# mission_sim/analysis/interplanetary/sf_design.py
# -*- coding: utf-8 -*-
"""Sims-Flanagan 式连续小推力（SEP）转移设计器（Ceres E1 能力沉淀件）。

来源：谷神星探测专题 E1（``工具/s4_lt_design.py``）。算法、离散口径与判据不变
（对拍见 ``tests/test_interplanetary_low_thrust.py``：与岛内机读结果逐位一致）。

模型（与专题岛既有 LT 工作同源同口径）
--------------------------------------
* **动力学**：日心二体（μ☉）＋**分段冲量**；段间传播 = 通用变量 Kepler
  （``low_thrust.kep``），与 ``two_body.Ephemeris`` 同一星历口径。
* **推力约束**：节点 k 的冲量上限 ``cap_k = a_max(r_k, m_k)·min(Δt_{k-1}, Δt_k)``；
  ``a_max`` 两种口径：``sep``（功率-推力模型 ``low_thrust.SEPModel``，本岛新增）与
  ``const``（与专题岛 C1/E6 的 LT 工作同口径，常数 a）。段数相关的 cap 离散伪影在专题岛
  ``E6_N10_SF多起点.md`` §三-4 量化（~8.6%），本件沿用该口径并在结果里标注 N 与 a。
* **边界条件**：
  - 起点 ``depart_earth``：``r=r_E``、``v=v_E+√C3·û(α,β)``（C3 由运载给定、方向自由）；
    ``start_body``：``v=v_body``（mode B，全部 Δv 由推力给）。
  - 中间天体 ``flyby``：只匹配位置（3 等式）；``rendezvous``：位置＋速度（6 等式）。
  - 终点 ``arrive_earth``：位置匹配 ＋ ``v∞ ≤ vinf_max`` 不等式（SRC 直接再入，Vinf<6 km/s）。
* **求解**：SLSQP 直接法（scipy）＋多起点（切向螺旋打靶 / 脉冲-Lambert / 固定种子随机扰动），
  L-BFGS-B 罚函数版作热启动。结果带收敛历史摘要与随机种子。

**目标函数口径（源件原样）**：``Prob.obj`` = 节点冲量**分量绝对值之和**（L1），而
``counters["total"]`` = 节点冲量**模长之和**（物理 Σ|Δv|）；两者在一般解下不相等。
``sf_solve`` 返回 dict **同时命名**两者（``obj_L1``／``dv_L2``，``total`` 键保留向后兼容）。

**内向转移（水星/金星）补丁**（源件面向谷神星外向转移，内向长弧上需下列开关）：
* 出发方向界可注入：``Prob(dep_alpha_half=…)``／``_bounds(..., dep_alpha_half=…)``
  （默认 0.8 rad＝外向口径；内向转移可取 π ⇒ 出发方向自由，运载 √C3 得沿近逆行抵扣）；
* 可行性前级默认开：``sf_solve(feas_stage=True)`` 先 ``least_squares`` 压端点约束残差
  （只在改善时采用），再进罚函数＋SLSQP 流水线；``feas_metric`` 可选残差口径
  （``"eq"``＝源件口径；``"resid"``＝按 ``resid_km`` 统一量纲并含 cap/v∞ 违约量——
  位置与速度混合约束（交会）的必备口径）；
* **非等分分段**：``segs`` 元素可为段数（``int``，旧口径）或时长/分数权重序列；
  ``Prob(end_dense=k)`` 把末腿末段再均分为 k 段（长弧末端加密）。

用法::

  python3 -m mission_sim.analysis.interplanetary.sf_design --self-test
  python3 -m mission_sim.analysis.interplanetary.sf_design --sr1 --quick
  python3 -m mission_sim.analysis.interplanetary.sf_design --variants --quick
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# 线程上限（在 import numpy/scipy **之前**设）：本件为小数组运算，多线程 BLAS 只增同步开销
# （专题岛 E1 实测 CPU/墙钟 = 24×、墙钟被拖慢 ~20×）。该经验对仿真线可复用。
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import minimize

from mission_sim.analysis.interplanetary import low_thrust as LT

AU = LT.AU_KM
DAY = LT.DAY
M0_KG = (
    4669.0  # 专题岛 mission.yaml point_design.mass.wet_kg（干 2138＋Xe 1320＋肼 1211）
)
ISP_S = 4000.0  # mission.yaml point_design.dv.isp_s（NEXT 实值）
SR1_C3 = 13.4  # 常量.yaml mission_references.pmcs_003.c3
SR1_TOF_D = 2376.0  # 动画/场景几何.json 航段天数.发射→抵达
SR1_RET_D = 1719.0  # 同上.离场→返回
SR1_DATES = {
    "launch": "2030-12-15",
    "arrive": "2037-06-17",
    "depart": "2038-10-30",
    "back": "2043-07-15",
}
SEED_DEFAULT = 20261015

# 出发方向角半界 [rad]（默认＝谷神星线**外向转移**口径）：α∈α_E±DEP_ALPHA_HALF。
# **内向转移**（水星/金星）的最优出发渐近线近似逆行，调用方需注入更大半界
# （可取 π ⇒ 出发方向自由），否则运载 √C3 的抵扣方向被封在 0.8 rad 内。
DEP_ALPHA_HALF = 0.8
DEP_BETA_HALF = 0.35  # β 的 ±20° 为**工程界**【假设】（未登记实际 C3 赤纬包络）
FEAS_MAX_NFEV = 2000  # 可行性前级（least_squares）默认 nfev 上限


# ================================================================ 问题描述
def _split_legs(
    nodes: Sequence[float], segs: Sequence[Any], end_dense: int = 1
) -> Tuple[List[int], List[List[float]], List[bool]]:
    """把每腿的 ``segs`` 规格解算为（段数、段时长分数、是否等分）。

    ``segs[j]`` 两种写法（**旧写法行为逐位保留**）：
      * ``int n``：该腿**等分**为 n 段（源件口径）；
      * ``[1/3, 1/3, 1/3]`` 形（长度＝段数）：按**分数/时长权重**分段（权重和不必为 1，
        内部按和归一）⇒ 支持非等分分段。
    ``end_dense=k``（k≥2）把**末腿的末段**再均分为 k 段（端点加密：长弧末端打靶对小 Δt
    敏感），总时长不变、末腿段数由 n 变 n−1+k。
    """
    assert len(segs) == len(nodes) - 1, "segs 长度须等于腿数（len(nodes)−1）"
    counts: List[int] = []
    fracs: List[List[float]] = []
    equal: List[bool] = []
    for spec in segs:
        if isinstance(spec, (int, np.integer)):
            n = int(spec)
            if n < 1:
                raise ValueError("每腿段数须 ≥1，得到 %r" % (spec,))
            counts.append(n)
            fracs.append([1.0 / n] * n)
            equal.append(True)
        else:
            w = [float(q) for q in spec]
            if not w or any(q <= 0.0 for q in w):
                raise ValueError("分段权重须为非空正数序列，得到 %r" % (spec,))
            s = math.fsum(w)
            counts.append(len(w))
            fracs.append([q / s for q in w])
            equal.append(False)
    k = int(end_dense)
    if k > 1:
        last = fracs[-1]
        counts[-1] = counts[-1] - 1 + k
        fracs[-1] = last[:-1] + [last[-1] / k] * k
        equal[-1] = False
    return counts, fracs, equal


class Prob:
    """一条（或一条链的）小推力转移问题。

    Args:
        nodes: 节点偏日数（离 J2000），长度 M+1（M＝腿数）。
        segs: 每腿段数，长度 M。元素为 ``int n``（该腿等分 n 段，旧口径）或
            **非等分**权重序列（如 ``[1.0, 2.0, 4.0]``：3 段，时长比 1:2:4；和不必为 1）。
        bodies: 天体键，长度 M+1。
        kinds: 长度 M+1；元素 0 为 ``depart_earth`` / ``start_body``，中间为
            ``flyby`` / ``rendezvous``，末元素为终点类型。
        model: ``SEPModel`` / ``ConstAModel``。
        m0: 起始质量 [kg]。
        c3: 运载注入 C3 [km²/s²]（``depart_earth`` 时必需）。
        vinf_max: ``arrive_earth`` 的地球再入 v∞ 上限 [km/s]。
        name: 问题名（进结果与 ``fmt()``）。
        meta: 附加元数据。
        dep_alpha_half / dep_beta_half: 出发方向角半界 [rad]（见 :func:`_bounds`；
            默认 0.8/0.35 为**外向转移**口径；内向转移可注入 ``math.pi``）。
        end_dense: ``k≥2`` 时把**末腿的末段**再均分为 k 段（端点加密）。
    """

    def __init__(
        self,
        nodes: Sequence[float],
        segs: Sequence[Any],
        bodies: Sequence[str],
        kinds: Sequence[str],
        model: Any,
        m0: float = M0_KG,
        c3: Optional[float] = None,
        vinf_max: Optional[float] = None,
        name: str = "",
        meta: Optional[Mapping[str, Any]] = None,
        dep_alpha_half: float = DEP_ALPHA_HALF,
        dep_beta_half: float = DEP_BETA_HALF,
        end_dense: int = 1,
    ) -> None:
        self.nodes = list(map(float, nodes))
        self.bodies = list(bodies)
        self.kinds = list(kinds)
        self.model = model
        self.m0 = float(m0)
        self.c3 = c3
        self.vinf_max = vinf_max
        self.name = name
        self.meta = dict(meta or {})
        self.dep_alpha_half = float(dep_alpha_half)
        self.dep_beta_half = float(dep_beta_half)
        self.end_dense = int(end_dense)
        assert len(segs) == len(self.nodes) - 1 == len(self.bodies) - 1
        self.leg_segs, self.leg_fracs, self.leg_equal = _split_legs(
            self.nodes, segs, end_dense=self.end_dense
        )
        self.segs = list(self.leg_segs)  # 每腿**段数**（int，旧语义不变）
        dtl: List[float] = []
        for j in range(len(self.nodes) - 1):
            dur = self.nodes[j + 1] - self.nodes[j]
            if self.leg_equal[j]:  # 等分：保持源件逐位口径
                dtl += [dur / self.segs[j]] * self.segs[j]
            else:
                dtl += [dur * f for f in self.leg_fracs[j]]
        self.dt = dtl
        self.N = sum(self.segs)
        self.nimp = self.N - 1
        self.bstart = self.kinds[0] == "depart_earth"
        self.nv = 3 * self.nimp + (2 if self.bstart else 0)
        # 遭遇节点：段索引（1..N）
        self.enc: List[Tuple[int, str, str]] = []
        for j in range(1, len(self.nodes)):
            self.enc.append((sum(self.segs[:j]), self.kinds[j], self.bodies[j]))
        # 各节点天体状态（星历，只在建问题时取一次）
        self.body_state: Dict[int, Tuple[LT.Vec3, LT.Vec3]] = {}
        for j, b in enumerate(self.bodies):
            if LT.is_body(b):
                self.body_state[j] = LT.state_off(b, self.nodes[j])

    # ---------------------------------------------------------- 初始条件
    def ic(
        self, alpha: Optional[float] = None, beta: Optional[float] = None
    ) -> Tuple[LT.Vec3, LT.Vec3]:
        if self.bstart:
            r, v = self.body_state[0]
            ub = [
                math.cos(beta) * math.cos(alpha),
                math.cos(beta) * math.sin(alpha),
                math.sin(beta),
            ]
            vi = math.sqrt(self.c3)
            return list(r), [v[0] + vi * ub[0], v[1] + vi * ub[1], v[2] + vi * ub[2]]
        r, v = self.body_state[0]
        return list(r), list(v)

    def alpha0(self) -> float:
        _, v = self.body_state[0]
        return math.atan2(v[1], v[0])

    # ---------------------------------------------------------- 传播
    def propagate(self, x: Sequence[float]) -> Dict[str, Any]:
        """由设计向量 x 前向传播，返回 dict（节点位置/速度、冲量、上限、质量）。"""
        x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=1e4, neginf=-1e4)
        if self.bstart:
            al, be = float(x[0]), float(x[1])
            dv = np.asarray(x[2:], float).reshape(self.nimp, 3)
        else:
            al, be = None, None
            dv = np.asarray(x, float).reshape(self.nimp, 3)
        r, v = self.ic(al, be)
        dt = self.dt
        ve = self.model.ve
        mags = np.zeros(self.nimp)
        caps = np.zeros(self.nimp)
        rnode = [list(r)]
        vnode = [list(v)]
        mass_node = [self.m0]
        dvacc = 0.0
        m = self.m0
        for k in range(1, self.N + 1):
            j = k - 2
            if 0 <= j < self.nimp:
                d = dv[j]
                v = [v[0] + d[0], v[1] + d[1], v[2] + d[2]]
                mg = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                mags[j] = mg
                dvacc += mg
                m = self.m0 * math.exp(-dvacc / ve)
            r, v = LT.kep(r, v, dt[k - 1])
            rnode.append(list(r))
            vnode.append(list(v))
            mass_node.append(m)
            if 1 <= k <= self.nimp:
                a = self.model.a_km_s2(LT.vnorm(r), m)
                caps[k - 1] = a * min(dt[k - 1], dt[k]) * DAY
        return dict(
            r=[rnode[k] for k, _, _ in self.enc],
            v=[vnode[k] for k, _, _ in self.enc],
            rnode=rnode,
            vnode=vnode,
            dv=dv,
            mags=mags,
            caps=caps,
            mass_node=mass_node,
            total=float(mags.sum()),
        )

    # ---------------------------------------------------------- 等式约束（/AU 与 km/s）
    def eq_cons(self, x: Sequence[float]) -> np.ndarray:
        tr = self.propagate(x)
        out: List[float] = []
        for i, (_k, kind, _body) in enumerate(self.enc):
            rb, vb = self.body_state[i + 1]
            dr = LT.vsub(tr["r"][i], rb)
            out += [dr[0] / AU, dr[1] / AU, dr[2] / AU]
            if kind == "rendezvous":
                dv = LT.vsub(tr["v"][i], vb)
                out += dv
        return np.array(out)

    def ineq_cons(self, x: Sequence[float]) -> np.ndarray:
        tr = self.propagate(x)
        out = list(tr["caps"] ** 2 - (tr["dv"] ** 2).sum(axis=1))
        for i, (_k, kind, _body) in enumerate(self.enc):
            if kind == "arrive_earth" and self.vinf_max:
                _, vb = self.body_state[i + 1]
                d = LT.vsub(tr["v"][i], vb)
                out.append(self.vinf_max**2 - LT.vdot(d, d))
        return np.array(out)

    def obj(self, x: Sequence[float]) -> float:
        if self.bstart:
            return float(np.abs(np.asarray(x[2:], float).reshape(self.nimp, 3)).sum())
        return float(np.abs(np.asarray(x, float).reshape(self.nimp, 3)).sum())

    # ---------------------------------------------------------- 约束残差（km / km·s⁻¹）
    def resid_km(self, x: Sequence[float]) -> float:
        tr = self.propagate(x)
        worst = 0.0
        for i, (_k, kind, _body) in enumerate(self.enc):
            rb, vb = self.body_state[i + 1]
            worst = max(worst, LT.vnorm(LT.vsub(tr["r"][i], rb)))
            if kind == "rendezvous":
                worst = max(worst, LT.vnorm(LT.vsub(tr["v"][i], vb)) * 1e3)
        return worst

    def counters(self, x: Sequence[float]) -> Dict[str, Any]:
        """结果摘要：遭遇点 v∞、各腿 Δv 分项、触界统计、加速度利用率。"""
        tr = self.propagate(x)
        enc = []
        for i, (_k, kind, body) in enumerate(self.enc):
            rb, vb = self.body_state[i + 1]
            dv = LT.vsub(tr["v"][i], vb)
            enc.append(
                dict(
                    node=i + 1,
                    kind=kind,
                    body=body,
                    t_off=self.nodes[i + 1],
                    date=LT.ymd(self.nodes[i + 1]),
                    r_miss_km=LT.vnorm(LT.vsub(tr["r"][i], rb)),
                    v_miss_kms=LT.vnorm(dv),
                    vinf_kms=LT.vnorm(dv),
                )
            )
        legs = []
        off = 0
        for j, n in enumerate(self.segs):
            idx = list(range(max(0, off - 1), off + n - 1))
            idx = [q for q in idx if 0 <= q < self.nimp]
            legs.append(
                dict(
                    leg=j + 1,
                    from_body=self.bodies[j],
                    to_body=self.bodies[j + 1],
                    n_seg=n,
                    dv=float(tr["mags"][idx].sum()) if idx else 0.0,
                    dv_parts=[
                        float(tr["mags"][q])
                        for q in range(off, min(off + n, self.nimp))
                    ],
                )
            )
            off += n
        binding = int(np.sum(tr["mags"] > tr["caps"] * (1.0 - 1e-6)))
        avail_full = 0.0
        for k in range(1, self.N):
            m = tr["mass_node"][k]
            avail_full += (
                self.model.a_km_s2(LT.vnorm(tr["rnode"][k]), m) * self.dt[k - 1] * DAY
            )
        return dict(
            total=tr["total"],
            enc=enc,
            legs=legs,
            n_binding=binding,
            dv_sum_cap=float(np.sum(tr["caps"])),
            dv_sum_avail_full=avail_full,
            util=tr["total"] / avail_full if avail_full else float("nan"),
            dv_parts=[float(v) for v in tr["mags"]],
            caps=[float(v) for v in tr["caps"]],
            mass_final=tr["mass_node"][-1],
        )

    def fmt(self) -> str:
        return "%s：%s" % (
            self.name,
            " → ".join(
                ["%s@%s" % (self.bodies[0], LT.ymd(self.nodes[0]))]
                + [
                    "%s(%s)@%s" % (self.bodies[j], self.kinds[j], LT.ymd(self.nodes[j]))
                    for j in range(1, len(self.nodes))
                ]
            ),
        )


# ================================================================ 脉冲下界（Lambert）
def pulse_bound(prob: Prob, nmax: int = 3, nscan: int = 300) -> Dict[str, Any]:
    """同日期下**脉冲（Lambert）参照值**：逐腿独立解（含多圈 ≤nmax），取最优。

    dep：起点为 ``depart_earth`` 时 ``max(0, |v1_L − v_E| − √C3)``（运载注入抵扣），
         否则 ``|v1_L − v_body|``。
    joint：上一腿 v2_L 与下一腿 v1_L 之差。
    arr：终点 ``rendezvous`` 记 ``|v2_L − v_body|``；``flyby`` / ``arrive_earth`` 不记。

    **口径警告**：固定日期上 2 冲量弹道解严重病态（专题岛 ``动画/README.md`` 已记
    "2376 d 的直飞解给出 C3≈664，荒谬"）⇒ 本函数结果**不是低推力的严格下界**，
    只作"同日期 2 冲量参照"；量级判据用 :func:`spiral_scale` 与 :func:`pulse_bound_freetime`。
    """
    tot = 0.0
    parts: List[Dict[str, Any]] = []
    v2_prev: Optional[LT.Vec3] = None
    ok = True
    for j in range(len(prob.nodes) - 1):
        r1, _ = LT.state_off(prob.bodies[j], prob.nodes[j])
        r2, _ = LT.state_off(prob.bodies[j + 1], prob.nodes[j + 1])
        tof = prob.nodes[j + 1] - prob.nodes[j]
        sols = LT.lambert_multi(r1, r2, tof, nmax=nmax, nscan=nscan)
        if not sols:
            ok = False
            break
        best: Optional[Tuple[float, float, float, Mapping[str, Any]]] = None
        for s in sols:
            if v2_prev is not None:
                dep = LT.vnorm(LT.vsub(s["v1"], v2_prev))
            else:
                _, vb = prob.body_state[0]
                d0 = LT.vnorm(LT.vsub(s["v1"], vb))
                dep = max(0.0, d0 - math.sqrt(prob.c3)) if prob.bstart else d0
            arr = 0.0
            if prob.kinds[j + 1] == "rendezvous":
                _, vb2 = prob.body_state[j + 1]
                arr = LT.vnorm(LT.vsub(s["v2"], vb2))
            t = dep + arr
            if best is None or t < best[0]:
                best = (t, dep, arr, s)
        assert best is not None
        tot += best[0]
        parts.append(
            dict(
                leg=j + 1,
                dv=best[0],
                dep=best[1],
                arr=best[2],
                rev=best[3]["rev"],
                branch=best[3]["branch"],
            )
        )
        v2_prev = best[3]["v2"]
    return dict(
        total=tot,
        parts=parts,
        ok=ok,
        nmax=nmax,
        note=(
            "同日期 2 冲量（多圈 Lambert ≤%d）参照值；固定日期上此参考病态，"
            "见 docstring 警告。" % nmax
        ),
    )


def pulse_bound_freetime(
    t0_off: float,
    b_from: str,
    b_to: str,
    tof_grid: Sequence[float],
    c3: Optional[float] = None,
    arrival_kind: str = "rendezvous",
    nmax: int = 1,
    nscan: int = 200,
) -> Optional[Dict[str, Any]]:
    """**自由到达日**的 2 冲量参照：在 TOF 网格上取最优（量级判据用）。

    dep = ``max(0,|v1_L−v_from(t0)|−√C3)``（起点为地球时）或 ``|v1_L−v_from(t0)|``；
    arr = ``rendezvous`` 时 ``|v2_L−v_to(t1)|``，否则 0。空网格返回 None。
    """
    best: Optional[Dict[str, Any]] = None
    for tof in tof_grid:
        r1, v1b = LT.state_off(b_from, t0_off)
        r2, v2b = LT.state_off(b_to, t0_off + tof)
        for s in LT.lambert_multi(r1, r2, tof, nmax=nmax, nscan=nscan):
            d0 = LT.vnorm(LT.vsub(s["v1"], v1b))
            if c3 is not None and b_from == "earth":
                d0 = max(0.0, d0 - math.sqrt(c3))
            arr = (
                LT.vnorm(LT.vsub(s["v2"], v2b)) if arrival_kind == "rendezvous" else 0.0
            )
            tot = d0 + arr
            if best is None or tot < best["total"]:
                best = dict(
                    total=tot,
                    dep=d0,
                    arr=arr,
                    tof_d=float(tof),
                    t1=LT.ymd(t0_off + tof),
                    rev=s["rev"],
                    branch=s["branch"],
                )
    return best


# ================================================================ 解析螺旋标尺（Edelbaum 类比）
def spiral_scale(
    r1_au: float = 1.0, r2_au: Optional[float] = None, c3: Optional[float] = None
) -> Dict[str, Any]:
    """共面圆轨道**低推力螺旋**解析标尺（Edelbaum 类比）。

    * 单程螺旋 Δv = ``|v_c(r1) − v_c(r2)|``；
    * 往返 ＝ 2×单程；扣运载注入 √C3 ⇒ "纯 SEP 螺旋"上标尺；
    * Hohmann 型脉冲（含捕获/出发）给下标尺。
    """
    r2_au = r2_au or LT.BODIES["ceres"]["a_au"]
    r1, r2 = r1_au * AU, r2_au * AU
    vc1, vc2 = math.sqrt(LT.MU_SUN / r1), math.sqrt(LT.MU_SUN / r2)
    at = 0.5 * (r1 + r2)
    vt1 = math.sqrt(LT.MU_SUN * (2.0 / r1 - 1.0 / at))
    vt2 = math.sqrt(LT.MU_SUN * (2.0 / r2 - 1.0 / at))
    out = dict(
        r1_au=r1_au,
        r2_au=r2_au,
        vc1=vc1,
        vc2=vc2,
        spiral_one_way=abs(vc1 - vc2),
        spiral_round_trip=2 * abs(vc1 - vc2),
    )
    out["spiral_round_trip_minus_c3"] = out["spiral_round_trip"] - math.sqrt(
        c3 or SR1_C3
    )
    out["hohmann_dep"] = vt1 - vc1
    out["hohmann_arr"] = vc2 - vt2
    out["hohmann_round_trip"] = 2 * (
        out["hohmann_dep"] + out["hohmann_arr"]
    ) - math.sqrt(c3 or SR1_C3)
    out["note"] = (
        "螺旋＝|v_c1−v_c2|（单程）；往返×2 后扣运载注入 √C3 ⇒ 纯 SEP 螺旋标尺（上标尺）；"
        "Hohmann 型脉冲给下标尺。"
    )
    return out


# ================================================================ 初值：切向螺旋打靶
def _tangential_trj(
    prob: Prob,
    alpha: float,
    beta: float,
    sched: Sequence[Sequence[float]],
    scale: float = 1.0,
) -> Tuple[np.ndarray, List[LT.Vec3]]:
    """按"逐节点切向冲量＋推力窗"前向传播。

    ``sched``：每条腿一个 ``(f0, f1, lam)``——腿内归一化时间窗 [f0,f1] 内以 lam
    （−1..1，符号＝顺/逆行切向）乘 cap 施加冲量，窗外＝滑行。**推力窗是相位（phasing）
    的关键自由度**。
    """
    n = prob.N
    dt = prob.dt
    r, v = prob.ic(alpha, beta)
    leg_of: List[int] = []
    frac_of: List[float] = []
    for j in range(len(prob.segs)):
        if prob.leg_equal[j]:  # 等分：保持源件逐位口径
            nseg = prob.segs[j]
            for i in range(nseg):
                leg_of.append(j)
                frac_of.append((i + 0.5) / nseg)
        else:  # 非等分：段中点的累计归一化时间
            cum = 0.0
            for f in prob.leg_fracs[j]:
                leg_of.append(j)
                frac_of.append(cum + 0.5 * f)
                cum += f
    dv = np.zeros((prob.nimp, 3))
    ve = prob.model.ve
    dvacc = 0.0
    m = prob.m0
    rn = [list(r)]
    for k in range(1, n + 1):
        j = k - 2
        if 0 <= j < prob.nimp:
            leg = leg_of[j]
            f0, f1, lam = sched[min(leg, len(sched) - 1)]
            lam = max(-1.0, min(1.0, lam))
            on = f0 <= frac_of[j] <= f1
            u = LT.vunit(v)
            a = prob.model.a_km_s2(LT.vnorm(rn[-1]), m)
            cap = a * min(dt[j], dt[j + 1]) * DAY
            mag = lam * cap * scale if on else 0.0
            dv[j] = [u[0] * mag, u[1] * mag, u[2] * mag]
            v = [v[0] + dv[j][0], v[1] + dv[j][1], v[2] + dv[j][2]]
            dvacc += abs(mag)
            m = prob.m0 * math.exp(-dvacc / ve)
        r, v = LT.kep(r, v, dt[k - 1])
        rn.append(list(r))
    return dv, rn


def init_tangential(
    prob: Prob, seed: int = SEED_DEFAULT, n_lams: int = 4, verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """切向螺旋打靶初值（两套调度，多起点取优），返回 (x0, 残差)。

    * ``mode='lam'``（起点为 ``depart_earth``）：每腿单一节流 λ∈[−1,1]。
    * ``mode='win'``（起点为 ``start_body``＝mode B）：每腿 ``(f0,f1,λ)`` 推力窗。
    """
    rng = np.random.default_rng(seed)
    m_legs = len(prob.segs)
    win_mode = (not prob.bstart) or prob.meta.get("init") == "win"
    tgt = []
    for j in range(1, len(prob.nodes)):
        rb, vb = prob.body_state[j]
        tgt.append((j, prob.kinds[j], rb, vb))

    def sched_of(p: Sequence[float]) -> List[Tuple[float, float, float]]:
        if win_mode:
            return [
                tuple(np.clip(p[2 + 3 * q : 5 + 3 * q], [-1, -1, -1], [2, 2, 1]))
                for q in range(m_legs)
            ]
        return [(0.0, 1.0, float(np.clip(p[2 + q], -1.0, 1.0))) for q in range(m_legs)]

    def err(p: Sequence[float]) -> float:
        alpha, beta = p[0], p[1]
        _, rn = _tangential_trj(prob, alpha, beta, sched_of(p))
        e = 0.0
        for j, _kind, rb, _vb in tgt:
            k = sum(prob.segs[:j])
            e += (LT.vnorm(LT.vsub(rn[k], rb)) / 1e7) ** 2
        return math.sqrt(e)

    a0 = prob.alpha0()
    best: Any = None
    cands: List[List[float]] = []
    if win_mode:
        for lam in (1.0, -1.0):
            for f0 in (0.0, 0.25, 0.5):
                cands.append(
                    [a0, 0.0] + [x for _ in range(m_legs) for x in (f0, 1.0, lam)]
                )
    for l0 in np.linspace(-1.0, 1.0, max(3, n_lams)):
        cands.append(
            [a0, 0.0]
            + (
                [x for _ in range(m_legs) for x in (0.0, 1.0, l0)]
                if win_mode
                else [l0] * m_legs
            )
        )
    for p0 in cands:
        res = minimize(
            err,
            np.array(p0, float),
            method="Nelder-Mead",
            options=dict(maxiter=2500, xatol=1e-6, fatol=1e-8),
        )
        if best is None or res.fun < best.fun:
            best = res
    for _ in range(4):  # 固定种子随机多起点
        if win_mode:
            tail = [
                x
                for _ in range(m_legs)
                for x in (
                    abs(rng.normal(0, 0.3)),
                    0.7 + abs(rng.normal(0, 0.3)),
                    rng.uniform(-1, 1),
                )
            ]
        else:
            tail = [rng.uniform(-1, 1) for _ in range(m_legs)]
        p0 = np.array([a0 + rng.normal(0, 0.25), rng.uniform(-0.3, 0.3)] + tail)
        res = minimize(
            err,
            p0,
            method="Nelder-Mead",
            options=dict(maxiter=2500, xatol=1e-6, fatol=1e-8),
        )
        if res.fun < best.fun:
            best = res
    if verbose:
        print(
            "    切向打靶初值残差 %.2f（%s：α=%.3f β=%.3f 调度 %s）"
            % (
                best.fun,
                "推力窗" if win_mode else "单节流",
                best.x[0],
                best.x[1],
                np.round(np.array(sched_of(best.x)).ravel(), 3),
            )
        )
    dv, _ = _tangential_trj(prob, best.x[0], best.x[1], sched_of(best.x))
    if prob.bstart:
        return np.concatenate([[best.x[0], best.x[1]], dv.reshape(-1)]), float(best.fun)
    return dv.reshape(-1), float(best.fun)


def init_pulse(prob: Prob) -> Optional[np.ndarray]:
    """脉冲（Lambert）初值：把 dep/joint 冲量放在对应节点，其余为零。"""
    x = np.zeros(prob.nv)
    v2_prev: Optional[LT.Vec3] = None
    legs = []
    for j in range(len(prob.nodes) - 1):
        r1, _ = LT.state_off(prob.bodies[j], prob.nodes[j])
        r2, _ = LT.state_off(prob.bodies[j + 1], prob.nodes[j + 1])
        sols = LT.lambert2(r1, r2, prob.nodes[j + 1] - prob.nodes[j])
        if not sols:
            return None
        best: Optional[Tuple[float, LT.Vec3, LT.Vec3]] = None
        for s in sols:
            v1 = s["v1"]
            if j == 0:
                _, vb = prob.body_state[0]
                d = LT.vnorm(LT.vsub(v1, vb))
                if prob.bstart:
                    d = max(0.0, d - math.sqrt(prob.c3))
            else:
                d = LT.vnorm(LT.vsub(v1, v2_prev))
            if best is None or d < best[0]:
                best = (d, v1, s["v2"])
        assert best is not None
        legs.append(best)
        v2_prev = best[2]
    off = 2 if prob.bstart else 0
    d0 = LT.vsub(legs[0][1], prob.body_state[0][1])
    if prob.bstart:
        x[0] = prob.alpha0()
        x[1] = 0.0
    k1 = prob.segs[0] - 1
    if 0 <= k1 < prob.nimp:
        x[off + 3 * k1 : off + 3 * k1 + 3] = d0
    for m in range(1, len(prob.segs)):
        dj = LT.vsub(legs[m][1], legs[m - 1][2])
        idx = sum(prob.segs[:m]) - 1
        if 0 <= idx < prob.nimp:
            x[off + 3 * idx : off + 3 * idx + 3] = dj
    return x


# ================================================================ 求解器
def _clip_caps(prob: Prob, x: Sequence[float]) -> np.ndarray:
    """把超出 cap 的节点冲量按比例削到 cap（初值可行化）。"""
    tr = prob.propagate(x)
    dv = tr["dv"].copy()
    caps = tr["caps"]
    for i in range(prob.nimp):
        mg = float(np.linalg.norm(dv[i]))
        if caps[i] > 0 and mg > caps[i]:
            dv[i] *= caps[i] / mg
    if prob.bstart:
        return np.concatenate([np.asarray(x, float)[:2], dv.reshape(-1)])
    return dv.reshape(-1)


def _bounds(
    prob: Prob,
    x0: Sequence[float],
    dep_alpha_half: Optional[float] = None,
    dep_beta_half: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """设计变量边界。冲量分量 ∈ ±2·max(cap)；出发方向 α∈α_E±half、β∈±half。

    ``half`` 取 ``Prob.dep_alpha_half``／``Prob.dep_beta_half``（构造默认 0.8／0.35 rad
    ＝**外向转移**口径，谷神星线）；显式传入的同名参数优先。**内向转移**（水星/金星）的
    出发渐近线近似逆行、需 α 半界可取到 π（出发方向自由），否则运载 √C3 的抵扣方向被封死。
    β 的 ±0.35 rad（±20°）为**工程界**【假设】（专题岛未登记实际 C3 赤纬包络，见待裁定）。
    """
    tr = prob.propagate(_clip_caps(prob, x0))
    b = max(2.0 * float(np.max(tr["caps"])), 0.05)
    a0 = prob.alpha0()
    ah = float(getattr(prob, "dep_alpha_half", DEP_ALPHA_HALF))
    bh = float(getattr(prob, "dep_beta_half", DEP_BETA_HALF))
    if dep_alpha_half is not None:
        ah = float(dep_alpha_half)
    if dep_beta_half is not None:
        bh = float(dep_beta_half)
    lo: List[float] = [a0 - ah, -bh] if prob.bstart else []
    hi: List[float] = [a0 + ah, bh] if prob.bstart else []
    lo += [-b] * (3 * prob.nimp)
    hi += [b] * (3 * prob.nimp)
    return list(zip(lo, hi))


def feas_resid_fn(prob: Prob, metric: str = "eq") -> Any:
    """构造可行性前级的残差函数（``metric`` 决定量纲口径）。

    * ``"eq"``（默认，源件/D1 口径）：直接用 :meth:`Prob.eq_cons`
      （位置以 AU、速度以 km/s 计）。**位置项被 AU 归一化后远小于速度项**，
      故该口径在"位置＋速度"混合约束下会优先闭合速度、放弃位置。
    * ``"resid"``：按 :meth:`Prob.resid_km` 的同一口径统一量纲
      （位置 /1e3 km、速度 /1 km·s⁻¹，两者可直接比较），并**把违约的等式-伴随不等式
      一并最小化**：节点冲量超 cap 的正残差与 ``arrive_earth`` 的 v∞ 超越量（km/s）。
      ⇒ 前级目标＝"把 x 推到**可行集**"（等式闭合＋冲量受限＋v∞ 受限），而非只压位置。
      长弧＋混合约束问题（水星内向转移）上，该口径是收敛与否的分水岭。
    """
    if metric == "eq":
        return prob.eq_cons
    if metric != "resid":
        raise ValueError("feas metric 须为 'eq' 或 'resid'，得到 %r" % (metric,))

    def _fn(z: Sequence[float]) -> np.ndarray:
        tr = prob.propagate(z)
        out: List[float] = []
        for i, (_k, kind, _body) in enumerate(prob.enc):
            rb, vb = prob.body_state[i + 1]
            dr = LT.vsub(tr["r"][i], rb)
            out += [dr[0] / 1e3, dr[1] / 1e3, dr[2] / 1e3]
            if kind == "rendezvous":
                out += list(LT.vsub(tr["v"][i], vb))
            elif kind == "arrive_earth" and prob.vinf_max:
                dv = LT.vsub(tr["v"][i], vb)
                out.append(max(0.0, LT.vnorm(dv) - prob.vinf_max))
        ov = tr["mags"] - tr["caps"]
        out += [max(0.0, float(q)) for q in ov]
        return np.array(out)

    return _fn


def feas_prestage(
    prob: Prob,
    x0: Sequence[float],
    bnds: Optional[Sequence[Tuple[float, float]]] = None,
    max_nfev: int = FEAS_MAX_NFEV,
    keep_best: bool = True,
    metric: str = "eq",
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """**可行性前级**：在界内用 ``least_squares`` 最小化端点约束残差。

    长弧（7–10 年）问题上，SLSQP 直接从切向初值出发常停在残差 1e6–1e7 km；
    先做本前级把 x 推到约束流形附近，再交给 ``sf_solve`` 的精修段收口。
    ``keep_best=True``（默认）时，若前级未改善残差则**原样返回 x0**（不会变差）。
    ``metric`` 见 :func:`feas_resid_fn`（默认 ``"eq"``＝源件/D1 口径；
    ``"resid"``＝按 ``resid_km`` 统一量纲并含不等式违约量）。
    **注**：``bnds`` 缺省时由传入点的 cap 现算（与 ``sf_solve`` 同口径），故界会随传入点变化；
    需要"界不变"的对照请显式传 ``bnds``。
    返回 ``(x, resid_km, info)``。
    """
    from scipy.optimize import least_squares

    bnds = list(_bounds(prob, x0) if bnds is None else bnds)
    lo = np.array([q[0] for q in bnds], float)
    hi = np.array([q[1] for q in bnds], float)
    x_in = np.clip(np.asarray(x0, float), lo, hi)
    r_in = prob.resid_km(x_in)
    try:
        ls = least_squares(
            feas_resid_fn(prob, metric),
            x_in,
            bounds=(lo, hi),
            method="trf",
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=int(max_nfev),
            x_scale="jac",
        )
    except Exception as exc:  # noqa: BLE001
        return x_in, r_in, dict(
            ok=False,
            resid_in_km=r_in,
            resid_km=r_in,
            nfev=0,
            msg="%s: %s" % (exc.__class__.__name__, exc),
        )
    x_f = np.asarray(ls.x, float)
    r_f = prob.resid_km(x_f)
    info = dict(
        ok=True,
        resid_in_km=r_in,
        resid_km=r_f,
        nfev=int(ls.nfev),
        msg=str(ls.message)[:60],
        kept=r_f >= r_in if keep_best else False,
    )
    if keep_best and not (r_f < r_in):
        return x_in, r_in, info
    return x_f, r_f, info


def sf_solve(
    prob: Prob,
    x0: Optional[Sequence[float]] = None,
    maxiter: int = 250,
    ftol: float = 1e-10,
    seed: int = SEED_DEFAULT,
    verbose: bool = False,
    multistart: int = 3,
    feas_stage: bool = True,
    feas_max_nfev: int = FEAS_MAX_NFEV,
    feas_metric: str = "eq",
) -> Dict[str, Any]:
    """SLSQP 直接法＋罚函数热启动。返回结果 dict（含收敛历史摘要、随机种子）。

    ``feas_stage=True``（默认）时，先对本问题首起点做一次**可行性前级**
    （:func:`feas_prestage`：``least_squares`` 压端点约束残差，**只在改善时采用**），
    再进 L-BFGS 罚函数＋SLSQP 流水线。长弧问题上该前级对收敛与否起决定作用；
    关掉即退回源件流水线（供对拍与回归用）。

    ``feas_max_nfev``：前级的 ``least_squares`` nfev 上限。
    ``feas_metric``：前级残差口径（``"eq"`` 默认＝源件/D1 口径；``"resid"``＝按
    ``resid_km`` 统一量纲并含冲量/cap 与 v∞ 违约量，见 :func:`feas_resid_fn`）。

    返回 dict 的 Δv 口径**双命名**：``obj_L1``（＝``total``，节点冲量分量绝对值之和，
    优化目标）与 ``dv_L2``（物理 Σ|Δv|，节点冲量模长之和）；``total`` 键保留向后兼容。
    """
    hist: List[Dict[str, Any]] = []
    cons = [dict(type="eq", fun=prob.eq_cons), dict(type="ineq", fun=prob.ineq_cons)]
    starts: List[Tuple[str, np.ndarray]] = []
    t0s = time.time()
    xi, _resi = (
        init_tangential(prob, seed=seed, verbose=verbose) if x0 is None else (x0, None)
    )
    starts.append(("tangential", np.asarray(xi, float)))
    xp = init_pulse(prob)
    if xp is not None:
        starts.append(("pulse", _clip_caps(prob, np.asarray(xp, float))))
    rng = np.random.default_rng(seed)
    for q in range(max(0, multistart - len(starts))):
        base = starts[0][1].copy()
        base = base + rng.normal(0, 0.02 * max(1.0, np.abs(base).max()), base.shape)
        starts.append(("rand%d" % (q + 1), base))
    bnds = _bounds(prob, starts[0][1])
    feas_info: Optional[Dict[str, Any]] = None
    if feas_stage:
        x_f, r_f, feas_info = feas_prestage(
            prob,
            _clip_caps(prob, starts[0][1]),
            bnds=bnds,
            max_nfev=feas_max_nfev,
            metric=feas_metric,
        )
        if prob.resid_km(x_f) < prob.resid_km(_clip_caps(prob, starts[0][1])):
            starts[0] = ("tangential+feas", np.asarray(x_f, float))
    print(
        "    设计变量 %d（含初值 %d 个起点）｜冲量分量界 ±%.3f km/s%s"
        % (
            prob.nv,
            len(starts),
            bnds[-1][1],
            "" if not feas_stage else "｜前级 %.3e→%.3e km" % (
                feas_info["resid_in_km"],
                feas_info["resid_km"],
            ),
        )
    )

    results = []
    for tag, x0v in starts:
        x0v = _clip_caps(prob, np.asarray(x0v, float))
        r0_km = prob.resid_km(x0v)
        xw = x0v.copy()
        if r0_km > 100.0:
            for w in (1e2, 1e5, 1e8):

                def pen(z: Sequence[float], w: float = w) -> float:
                    tr = prob.propagate(z)
                    mags = tr["mags"]
                    s = float(np.sum(mags))
                    e = prob.eq_cons(z)
                    ov = np.maximum(0.0, mags / np.maximum(tr["caps"], 1e-12) - 1.0)
                    return s + w * float(np.dot(e, e)) + 1e5 * float(np.dot(ov, ov))

                r = minimize(
                    pen,
                    xw,
                    method="L-BFGS-B",
                    bounds=bnds,
                    options=dict(maxiter=150, ftol=1e-14, gtol=1e-12),
                )
                xw = r.x
        hist.append(
            dict(
                tag=tag,
                stage="lbfgs_penalty",
                obj=prob.obj(xw),
                resid=prob.resid_km(xw),
            )
        )
        # (2) SLSQP 精修
        try:
            r2 = minimize(
                prob.obj,
                xw,
                method="SLSQP",
                bounds=bnds,
                constraints=cons,
                options=dict(maxiter=maxiter, ftol=ftol, disp=False),
            )
            xs = r2.x
            hist.append(
                dict(
                    tag=tag,
                    stage="slsqp",
                    obj=prob.obj(xs),
                    resid=prob.resid_km(xs),
                    nfev=int(r2.nfev),
                    msg=str(r2.message)[:60],
                )
            )
        except Exception as exc:  # noqa: BLE001
            xs = xw
            hist.append(
                dict(
                    tag=tag,
                    stage="slsqp",
                    obj=prob.obj(xs),
                    resid=prob.resid_km(xs),
                    msg="%s: %s" % (exc.__class__.__name__, exc),
                )
            )
        # (3) 再热启动一轮（若残差还大）
        for it in range(2):
            if prob.resid_km(xs) < 5.0:
                break
            try:
                r3 = minimize(
                    prob.obj,
                    xs,
                    method="SLSQP",
                    bounds=bnds,
                    constraints=cons,
                    options=dict(maxiter=maxiter, ftol=ftol, disp=False),
                )
                xs = r3.x
                hist.append(
                    dict(
                        tag=tag,
                        stage="slsqp_rs%d" % (it + 1),
                        obj=prob.obj(xs),
                        resid=prob.resid_km(xs),
                        nfev=int(r3.nfev),
                    )
                )
            except Exception:  # noqa: BLE001
                break
        counters = prob.counters(xs)
        results.append(
            dict(
                tag=tag,
                x=[float(v) for v in xs],
                total=prob.obj(xs),
                resid_km=prob.resid_km(xs),
                counters=counters,
            )
        )
        if verbose:
            print(
                "    起点 %-10s ⇒ ΣΔv=%.4f km/s  残差 %.3e km"
                % (tag, results[-1]["total"], results[-1]["resid_km"])
            )
    results.sort(key=lambda r: (r["resid_km"] > 20.0, r["total"]))
    best = results[0]
    return dict(
        name=prob.name,
        fmt=prob.fmt(),
        N=prob.N,
        segs=prob.segs,
        nodes=[float(t) for t in prob.nodes],
        dates=[LT.ymd(t) for t in prob.nodes],
        bodies=prob.bodies,
        kinds=prob.kinds,
        model=prob.model.as_dict(),
        m0_kg=prob.m0,
        c3=prob.c3,
        vinf_max=prob.vinf_max,
        nv=prob.nv,
        seed=seed,
        total=best["total"],
        obj_L1=best["total"],
        dv_L2=best["counters"]["total"],
        resid_km=best["resid_km"],
        best_tag=best["tag"],
        x=best["x"],
        counters=best["counters"],
        feas_stage=bool(feas_stage),
        feas_metric=feas_metric,
        feas=feas_info,
        all_starts=[
            dict(
                tag=r["tag"],
                total=r["total"],
                obj_L1=r["total"],
                dv_L2=r["counters"]["total"],
                resid_km=r["resid_km"],
            )
            for r in results
        ],
        history=hist,
        seconds=time.time() - t0s,
    )


# ================================================================ 问题工厂
def sep_model(
    k_deg: float = 1.0, eta: float = 0.70, P0: float = 27500.0, isp: float = ISP_S
) -> LT.SEPModel:
    """SEP 功率-推力模型工厂（默认档见 :class:`low_thrust.SEPModel`）。"""
    return LT.SEPModel(P0_W=P0, isp_s=isp, eta=eta, k_deg=k_deg)


def prob_sr1_out(
    model: Optional[Any] = None,
    c3: float = SR1_C3,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    nseg: int = 30,
    m0: float = M0_KG,
) -> Prob:
    """SR-1 去程：地球（C3=13.4）→ 谷神星**交会**（SEP）。"""
    t0 = LT.dfe(2030, 12, 15) if t0 is None else t0
    t1 = t0 + SR1_TOF_D if t1 is None else t1
    return Prob(
        [t0, t1],
        [nseg],
        ["earth", "ceres"],
        ["depart_earth", "rendezvous"],
        model or sep_model(),
        m0=m0,
        c3=c3,
        name="SR-1 去程（地球 C3=%.1f → 谷神星交会）" % c3,
    )


def prob_sr1_ret(
    model: Optional[Any] = None,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    nseg: int = 24,
    m0: float = M0_KG,
    vinf_max: float = 6.0,
) -> Prob:
    """SR-1 返程：谷神星离场（mode B）→ 地球（位置匹配＋v∞ ≤ 6 km/s）。"""
    t0 = LT.dfe(2038, 10, 30) if t0 is None else t0
    t1 = t0 + SR1_RET_D if t1 is None else t1
    return Prob(
        [t0, t1],
        [nseg],
        ["ceres", "earth"],
        ["start_body", "arrive_earth"],
        model or sep_model(),
        m0=m0,
        vinf_max=vinf_max,
        name="SR-1 返程（谷神星离场 → 地球再入）",
    )


def prob_chain(
    bodies: Sequence[str],
    kinds: Sequence[str],
    years: Sequence[Sequence[int]],
    model: Optional[Any] = None,
    t0: Optional[float] = None,
    tfly: Optional[float] = None,
    t2: Optional[float] = None,
    nseg1: int = 20,
    nseg2: int = 20,
    c3: float = SR1_C3,
    m0: float = M0_KG,
    vinf_max: float = 6.0,
    name: str = "",
) -> Prob:
    """两腿链：``bodies=[b0,b1,b2]``，``kinds=[起点类型, 中间类型, 终点类型]``。"""
    if t0 is None:
        t0 = LT.dfe(*years[0])
    if tfly is None:
        tfly = LT.dfe(*years[1])
    if t2 is None:
        t2 = LT.dfe(*years[2])
    return Prob(
        [t0, tfly, t2],
        [nseg1, nseg2],
        bodies,
        kinds,
        model or sep_model(),
        m0=m0,
        c3=c3 if kinds[0] == "depart_earth" else None,
        vinf_max=vinf_max,
        name=name or "%s→%s→%s" % tuple(bodies),
    )


# ================================================================ 自检（源件四项）
def run_self_test(
    nseg_sr1: int = 14, nseg_ret: int = 12, quick: bool = True
) -> Dict[str, Any]:
    """四项自检（源件 E1-2③）：① 解析螺旋标尺；② SF vs 脉冲下界；③ 两跑逐位一致；
    ④ 质量-Δv 链闭合。返回 res dict（含 ``ok``）。

    注：SF 的**绝对**数值依赖 scipy 版本（本件不锚定绝对值），判据只取**关系**
    （两跑一致、SF ≥ 脉冲下界、质量链闭合）与解析标尺锚。
    """
    t_start = time.time()
    print("=" * 96)
    print("sf_design · self-test（源件 E1-2③：可复算＋双路对拍）")
    print("=" * 96)
    res: Dict[str, Any] = {}

    sp = spiral_scale()
    print("① 解析螺旋标尺（共面圆轨道 1.000 → %.3f AU）【算】" % sp["r2_au"])
    print(
        "   圆轨道速度 v_c1=%.3f、v_c2=%.3f km/s ⇒ 单程螺旋 |Δv_c|=%.3f km/s；往返 %.3f km/s"
        % (sp["vc1"], sp["vc2"], sp["spiral_one_way"], sp["spiral_round_trip"])
    )
    print(
        "   扣运载注入 √C3=√%.1f=%.3f km/s ⇒ **纯 SEP 螺旋上标尺 %.2f km/s**（往返）"
        % (SR1_C3, math.sqrt(SR1_C3), sp["spiral_round_trip_minus_c3"])
    )
    print(
        "   Hohmann 型脉冲（dep %.2f＋arr %.2f，往返扣 C3）⇒ **下标尺 %.2f km/s**"
        % (sp["hohmann_dep"], sp["hohmann_arr"], sp["hohmann_round_trip"])
    )
    res["spiral_scale"] = sp

    pb_out = pulse_bound(prob_sr1_out(nseg=4))
    pb_ret = pulse_bound(prob_sr1_ret(nseg=4))
    print(
        "   本方案日期上的**脉冲（Lambert）下界**：去程 %.3f ＋ 返程 %.3f = %.3f km/s"
        % (pb_out["total"], pb_ret["total"], pb_out["total"] + pb_ret["total"])
    )
    res["pulse_bound"] = dict(out=pb_out, ret=pb_ret)

    print("\n③ 两跑一致性（同一初值/种子，逐位比较）")
    p = prob_sr1_out(nseg=nseg_sr1)
    x0a, _ = init_tangential(p, seed=SEED_DEFAULT)
    ra = sf_solve(p, x0=x0a, maxiter=80, seed=SEED_DEFAULT, multistart=1)
    x0b, _ = init_tangential(p, seed=SEED_DEFAULT)
    rb = sf_solve(p, x0=x0b, maxiter=80, seed=SEED_DEFAULT, multistart=1)
    same = (ra["total"] == rb["total"]) and (ra["x"] == rb["x"])
    print(
        "   跑1 ΣΔv=%.10f km/s（残差 %.3e km）｜跑2 ΣΔv=%.10f km/s（残差 %.3e km）"
        % (ra["total"], ra["resid_km"], rb["total"], rb["resid_km"])
    )
    print("   逐位一致：%s" % ("是（==）" if same else "**否**"))
    res["two_runs_identical"] = bool(same)
    res["two_runs"] = dict(total_a=ra["total"], total_b=rb["total"], same=same)

    print("\n② 低推力（SF）vs 脉冲下界（去程，SF 应 ≥ 下界）")
    print(
        "   SF(N=%d) %.4f km/s ｜ 脉冲下界 %.4f km/s ｜ 乘子 ×%.3f"
        % (nseg_sr1, ra["total"], pb_out["total"], ra["total"] / pb_out["total"])
    )
    res["lt_vs_pulse"] = dict(
        sf=ra["total"], pulse=pb_out["total"], k=ra["total"] / pb_out["total"]
    )
    ok_ge = ra["total"] >= pb_out["total"] * (1.0 - 1e-3)
    print("   判据（SF ≥ 下界）：%s" % ("通过" if ok_ge else "**未通过**"))

    print("\n④ 质量-Δv 链闭合（火箭方程，m0=4669 kg、Isp=4000 s）")
    dv_a = ra["total"]
    mf, prop = LT.rocket_prop(M0_KG, dv_a, ISP_S)
    dv_1320 = LT.dv_from_prop(M0_KG, 1320.0, ISP_S)
    print("   去程 SF Δv=%.4f km/s ⇒ m_终=%.1f kg，推进剂 %.1f kg" % (dv_a, mf, prop))
    print(
        "   Xe 1320 kg ⇔ Δv=%.4f km/s（mission.yaml 记『Isp 4000 s 下实达 13.03』）"
        % dv_1320
    )
    print(
        "   往返 %.3f km/s ⇒ 推进剂 %.1f kg（m0=4669）"
        % (dv_a * 2, M0_KG - LT.rocket_prop(M0_KG, dv_a * 2, ISP_S)[0])
    )
    res["mass_chain"] = dict(
        dv_out=dv_a,
        m_final=mf,
        prop_out=prop,
        dv_at_1320kg=dv_1320,
        prop_roundtrip=M0_KG - LT.rocket_prop(M0_KG, dv_a * 2, ISP_S)[0],
    )
    res["ok"] = bool(same and ok_ge and abs(dv_1320 - 13.0343) < 1e-3)
    print(
        "\n自检合计 %.1f s ⇒ %s"
        % (time.time() - t_start, "通过" if res["ok"] else "**未通过**")
    )
    print("=" * 96)
    return res


# ================================================================ SR-1 基线复现
def run_sr1(
    quick: bool = False,
    model: Optional[Any] = None,
    seed: int = SEED_DEFAULT,
    out_json: Optional[str] = None,
) -> Dict[str, Any]:
    """SR-1 基线复现（SEP 往返）：去程＋返程求解；``out_json`` 非空时落盘。"""
    n1 = 16 if quick else 30
    n2 = 14 if quick else 24
    model = model or sep_model()
    t_start = time.time()
    print("=" * 96)
    print("SR-1 基线复现（SEP 往返；发射/抵达/离场/再入见 SR1_DATES）")
    print("=" * 96)
    out: Dict[str, Any] = dict(
        kind="sr1_baseline",
        model=model.as_dict(),
        m0_kg=M0_KG,
        isp_s=ISP_S,
        c3=SR1_C3,
        dates=dict(SR1_DATES),
        tof_out_d=SR1_TOF_D,
        tof_ret_d=SR1_RET_D,
        seed=seed,
        reference=dict(
            source="003/515（Castillo-Rogez et al. 2022, PSJ 3:41）＋ mission.yaml",
            dv_sep_km_s=14.0,
            c3=13.4,
            xe_kg=1320,
            wet_kg=4669,
            reserve_km_s=0.97,
        ),
    )
    p_out = prob_sr1_out(model, nseg=n1)
    p_ret = prob_sr1_ret(model, nseg=n2)
    pb_out = pulse_bound(p_out)
    pb_ret = pulse_bound(p_ret)
    print(
        "脉冲下界：去程 %.4f（dep %.3f + arr %.3f）｜返程 %.4f｜合计 %.4f km/s"
        % (
            pb_out["total"],
            pb_out["parts"][0]["dep"],
            pb_out["parts"][0]["arr"],
            pb_ret["total"],
            pb_out["total"] + pb_ret["total"],
        )
    )
    print("SF 求解：去程 N=%d …" % n1)
    r_out = sf_solve(p_out, seed=seed, verbose=True, maxiter=300)
    print(
        "   去程 ΣΔv=%.4f km/s  残差 %.3e km  触界 %d 段  用时 %.0f s"
        % (
            r_out["total"],
            r_out["resid_km"],
            r_out["counters"]["n_binding"],
            r_out["seconds"],
        )
    )
    print("SF 求解：返程 N=%d …" % n2)
    r_ret = sf_solve(p_ret, seed=seed, verbose=True, maxiter=300)
    print(
        "   返程 ΣΔv=%.4f km/s  残差 %.3e km  触界 %d 段  用时 %.0f s"
        % (
            r_ret["total"],
            r_ret["resid_km"],
            r_ret["counters"]["n_binding"],
            r_ret["seconds"],
        )
    )
    tot = r_out["total"] + r_ret["total"]
    mf, prop = LT.rocket_prop(M0_KG, tot, ISP_S)
    out.update(
        dict(
            pulse_bound=dict(
                out=pb_out, ret=pb_ret, total=pb_out["total"] + pb_ret["total"]
            ),
            leg_out=r_out,
            leg_ret=r_ret,
            dv_out_km_s=r_out["total"],
            dv_ret_km_s=r_ret["total"],
            dv_total_km_s=tot,
            k_vs_pulse=tot / (pb_out["total"] + pb_ret["total"]),
            propellant_kg=prop,
            m_final_kg=mf,
            xe_equiv_kg=M0_KG - LT.rocket_prop(M0_KG, tot, ISP_S)[0],
            seconds=time.time() - t_start,
        )
    )
    print(
        "\n★ SR-1 复现：SEP 往返 ΣΔv = %.3f km/s（去程 %.3f ＋ 返程 %.3f）｜参考 14.0±1 ⇒ %s"
        % (
            tot,
            r_out["total"],
            r_ret["total"],
            "在带内" if abs(tot - 14.0) <= 1.0 else "**带外**",
        )
    )
    print("   推进剂（Xe 等效）%.1f kg ｜参考 1320 kg（003/515 口径）" % prop)
    if out_json:
        import json

        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
        print("  → %s" % out_json)
    print("合计 %.0f s" % (time.time() - t_start))
    return out


# ================================================================ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SEP 连续小推力轨迹设计（Sims-Flanagan）")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sr1", action="store_true")
    ap.add_argument(
        "--variants", action="store_true", help="委托 flyby_chain 跑顺访变体"
    )
    ap.add_argument("--quick", action="store_true", help="少段数试跑")
    ap.add_argument("--kdeg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument(
        "--out-json", default=None, help="SR-1 基线结果 JSON 路径（空＝不写）"
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    model = sep_model(k_deg=a.kdeg)
    if a.self_test:
        return 0 if run_self_test().get("ok") else 1
    if a.sr1:
        run_sr1(quick=a.quick, model=model, seed=a.seed, out_json=a.out_json)
        return 0
    if a.variants:
        from mission_sim.analysis.interplanetary import flyby_chain as fc

        fc.run_variants(quick=a.quick, model=model, seed=a.seed)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
