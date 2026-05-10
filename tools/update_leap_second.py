#!/usr/bin/env python3
"""
工具：更新闰秒数据 (Leap Second Data Updater)
-----------------------------------------------
从巴黎天文台的官方文件 Leap_Second.dat 中获取最新闰秒数据，
并与本地缓存文件 mission_sim/core/spacetime/Leap_Second.dat 比较。
若有差异则自动更新本地文件。

用法：
    python tools/update_leap_second.py

依赖：
    仅使用 Python 标准库 (urllib, datetime, os, re, argparse)。
"""

import argparse
import datetime
import os
import re
import sys
import urllib.request
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------
REMOTE_URL = "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "mission_sim", "core", "spacetime", "Leap_Second.dat")

# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------
def fetch_remote_data() -> str:
    """下载远程 Leap_Second.dat 文件内容"""
    try:
        with urllib.request.urlopen(REMOTE_URL, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"下载 Leap_Second.dat 失败: {e}") from e

# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse_leap_second_data(text: str) -> List[Tuple[str, int]]:
    """
    解析 Leap_Second.dat 文本，返回 (日期字符串 YYYY-MM-DD, TAI-UTC 偏移) 列表。
    """
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # 匹配数字列：MJD 日 月 年 TAI-UTC
        match = re.match(r'\s*(\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if not match:
            continue
        day = int(match.group(2))
        month = int(match.group(3))
        year = int(match.group(4))
        tai_utc = int(match.group(5))
        date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
        date_str = date.strftime("%Y-%m-%d")
        events.append((date_str, tai_utc))
    events.sort(key=lambda x: x[0])
    return events

# ---------------------------------------------------------------------------
# 本地文件操作
# ---------------------------------------------------------------------------
def load_local_data() -> Tuple[str, List[Tuple[str, int]]]:
    """读取本地 Leap_Second.dat 文本并解析，若不存在则返回空字符串和空列表"""
    if not os.path.exists(LOCAL_PATH):
        return "", []
    with open(LOCAL_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    events = parse_leap_second_data(text)
    return text, events

def save_local_data(text: str) -> None:
    """将最新数据文本写入本地文件"""
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    with open(LOCAL_PATH, 'w', encoding='utf-8') as f:
        f.write(text)

# ---------------------------------------------------------------------------
# 比较与显示
# ---------------------------------------------------------------------------
def compare_events(local_events: List[Tuple[str, int]], remote_events: List[Tuple[str, int]]) -> bool:
    """
    打印两个事件列表的差异，返回 True 表示有差异，False 表示一致。
    """
    local_dict = dict(local_events)
    remote_dict = dict(remote_events)

    added = set(remote_dict) - set(local_dict)
    removed = set(local_dict) - set(remote_dict)
    changed = {
        date for date in (set(local_dict) & set(remote_dict))
        if local_dict[date] != remote_dict[date]
    }

    if not (added or removed or changed):
        print("✅ 本地数据与远程数据完全一致，无需更新。")
        return False

    print("\n⚠️  闰秒数据存在差异：")
    if added:
        print("  新增/未来日期:")
        for d in sorted(added):
            print(f"    + {d} (TAI-UTC={remote_dict[d]})")
    if removed:
        print("  远程丢失日期（不应出现）:")
        for d in sorted(removed):
            print(f"    - {d} (TAI-UTC={local_dict[d]})")
    if changed:
        print("  偏移量变更:")
        for d in sorted(changed):
            print(f"    ~ {d} : {local_dict[d]} -> {remote_dict[d]}")
    return True

# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="从巴黎天文台获取最新闰秒数据，并与本地文件进行比较。"
    )
    # 无参数，自动更新本地文件
    args = parser.parse_args()

    # 下载远程数据
    print("🌐 正在从巴黎天文台获取最新 Leap_Second.dat...")
    try:
        remote_text = fetch_remote_data()
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    remote_events = parse_leap_second_data(remote_text)

    # 加载本地数据
    print(f"📂 读取本地文件 {LOCAL_PATH}...")
    local_text, local_events = load_local_data()
    if not local_events:
        print("   本地文件不存在或为空，将创建新文件。")

    # 比较并显示差异
    has_diff = compare_events(local_events, remote_events)

    if has_diff:
        # 更新本地文件
        print("\n🔄 正在更新本地文件...")
        save_local_data(remote_text)
        print("   本地 Leap_Second.dat 已更新。")
    else:
        # 即使无差异，如果本地文件不存在，也要写入
        if not local_text:
            save_local_data(remote_text)
            print("\n   已将远程数据保存到本地文件。")

    print("\n✅ 检查完成。")

if __name__ == "__main__":
    main()
