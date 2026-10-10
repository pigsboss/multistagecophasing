# mission_sim/analysis/sensitivity.py
# -*- coding: utf-8 -*-
"""
敏感度/不确定性工具（G5⑤ 沉淀件）
====================================
参数化扫描 + 区间结论口径：对不确定输入（如大气密度 ±50%）批量重跑，
汇总指标区间；结论一律以区间形式给出（禁止点值外推，R-AS-4）。
"""
import numpy as np


def parameter_sweep(run_fn, parameter, values, metric=None):
    """参数扫描。

    Args:
        run_fn: callable({parameter: value}) -> dict(指标名 -> 数值/对象)
        parameter: 参数名（传入 run_fn 的字典键）
        values: 参数取值序列
        metric: 汇总的指标键；None 时取各结果中公共的首个数值键

    Returns: dict(parameter, values, n_runs, results, interval{min,max,metric})
    """
    values = [float(v) for v in values]
    results = []
    for v in values:
        out = dict(run_fn({parameter: v}))
        results.append(out)
    if metric is None:
        common = None
        for res in results:
            keys = {k for k, val in res.items()
                    if isinstance(val, (int, float, np.integer, np.floating))}
            common = keys if common is None else (common & keys)
        metric = sorted(common)[0] if common else None
    interval = {}
    if metric is not None:
        vals = [float(res[metric]) for res in results]
        interval = {"min": float(min(vals)), "max": float(max(vals)), "metric": metric}
    return {
        "parameter": parameter,
        "values": values,
        "n_runs": len(results),
        "results": results,
        "interval": interval,
    }
