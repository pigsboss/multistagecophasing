# mission_sim/utils/dynamics/events.py
# -*- coding: utf-8 -*-
"""
事件检测基础设施（G5③ 沉淀件）
================================
从采样轨迹中检测：近地点/远地点（局部极值）、阈值穿越（大气进出）、
接近事件（月球近距，距离低于阈值）。

约定：t 单调；states 形状 (N, 6)（SI 或无量纲均可，单位与 r_earth 一致）。
"""
import numpy as np


def extremum_events(t, values, mode="min"):
    """局部极值事件（含端点），三点抛物线精修。"""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    out = []
    n = v.size
    for i in range(1, n - 1):
        if mode == "min" and v[i] < v[i - 1] and v[i] <= v[i + 1]:
            out.append(i)
        elif mode == "max" and v[i] > v[i - 1] and v[i] >= v[i + 1]:
            out.append(i)
    if n >= 2:
        if mode == "min" and v[0] <= v[1]:
            out.insert(0, 0)
        elif mode == "max" and v[0] >= v[1]:
            out.insert(0, 0)
        if mode == "min" and v[-1] <= v[-2] and (not out or out[-1] != n - 1):
            out.append(n - 1)
        elif mode == "max" and v[-1] >= v[-2] and (not out or out[-1] != n - 1):
            out.append(n - 1)

    events = []
    for i in out:
        if 0 < i < n - 1:
            y0, y1, y2 = v[i - 1], v[i], v[i + 1]
            denom = y0 - 2.0 * y1 + y2
            dt = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
            h = t[i + 1] - t[i]
            t_ev = t[i] + dt * h
            v_ev = y1 - 0.25 * (y0 - y2) * dt
        else:
            t_ev, v_ev = t[i], v[i]
        events.append({"t": float(t_ev), "index": int(i), "value": float(v_ev)})
    return events


def perigee_events(t, states, r_earth=None):
    """近地点事件（半径极小值）；给 r_earth 时附加高度。"""
    r = np.linalg.norm(np.asarray(states, dtype=np.float64)[:, :3], axis=1)
    events = extremum_events(t, r, mode="min")
    if r_earth is not None:
        for e in events:
            e["altitude"] = e["value"] - float(r_earth)
    return events


def threshold_crossings(t, values, threshold, direction="any"):
    """阈值穿越（线性插值）；direction ∈ {any, up, down}。"""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64) - float(threshold)
    out = []
    for i in range(v.size - 1):
        if v[i] == 0.0:
            continue
        if v[i] * v[i + 1] < 0.0:
            frac = -v[i] / (v[i + 1] - v[i])
            t_cross = t[i] + frac * (t[i + 1] - t[i])
            d = "up" if v[i + 1] > v[i] else "down"
            if direction in ("any", d):
                out.append({"t": float(t_cross), "direction": d})
    return out


def atmosphere_events(t, states, h_threshold=120e3, r_earth=6378137.0):
    """大气阈值事件：返回 entries / exits / intervals。"""
    r = np.linalg.norm(np.asarray(states, dtype=np.float64)[:, :3], axis=1)
    alt = r - float(r_earth)
    crosses = threshold_crossings(t, alt, h_threshold)
    entries = [c for c in crosses if c["direction"] == "down"]
    exits = [c for c in crosses if c["direction"] == "up"]
    intervals = []
    for ent in entries:
        nxt = [e for e in exits if e["t"] > ent["t"]]
        if nxt:
            intervals.append((ent["t"], nxt[0]["t"]))
    return {"entries": entries, "exits": exits, "intervals": intervals}


def approach_events(t, distance, threshold):
    """接近事件：distance < threshold 的连续区间（含最小距离）。"""
    t = np.asarray(t, dtype=np.float64)
    d = np.asarray(distance, dtype=np.float64)
    below = d < float(threshold)
    out = []
    i = 0
    n = d.size
    while i < n:
        if below[i]:
            j = i
            while j + 1 < n and below[j + 1]:
                j += 1
            # 边界细化（线性；首末点位于域内时给出穿越时刻）
            t0 = t[i]
            if i > 0:
                f = (d[i - 1] - threshold) / (d[i - 1] - d[i])
                t0 = t[i - 1] + f * (t[i] - t[i - 1])
            t1 = t[j]
            if j + 1 < n:
                f = (d[j] - threshold) / (d[j] - d[j + 1])
                t1 = t[j] + f * (t[j + 1] - t[j])
            i_min = i + int(np.argmin(d[i:j + 1]))
            out.append({"t_start": float(t0), "t_end": float(t1),
                        "t_min": float(t[i_min]), "min_distance": float(d[i_min])})
            i = j + 1
        else:
            i += 1
    return out
