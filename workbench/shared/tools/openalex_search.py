#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAlex 文献检索小工具（MCPC 共享文献区）
==========================================

用途
----
为三方（设计开发 / 评审专家 / 质量管理）提供统一的文献检索入口。
检索结果可下载入库到共享文献区（pdfs/ 目录 + library.md 登记）。

用法示例
--------
  # 基本检索（表格输出）
  python3 openalex_search.py "formation flying space interferometry" --limit 10

  # 只看开放获取 + 2015 年以后
  python3 openalex_search.py "co-phasing control" --oa --from-year 2015

  # 下载 OA 论文 PDF 到共享文献区
  python3 openalex_search.py "space interferometry" --oa \
      --download-dir ../literature/pdfs --download-limit 2

  # JSON 输出（供脚本处理）
  python3 openalex_search.py "..." --json

说明
----
- 数据源：OpenAlex API（https://api.openalex.org），无需密钥；带 mailto 进入礼貌池。
- 下载仅针对标记为开放获取（best_oa_location.pdf_url）的论文；支持断点续传，失败自动重试。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request

API = "https://api.openalex.org/works"
SELECT_FIELDS = (
    "id,doi,title,publication_year,cited_by_count,"
    "open_access,best_oa_location,primary_location"
)
DEFAULT_MAILTO = "mcpc-workbench@example.com"  # 仅用于进入 OpenAlex 礼貌池，可自行替换
USER_AGENT = "MCPC-workbench/0.1 (openalex literature screening)"


def build_url(query: str, oa: bool, from_year: int, limit: int, mailto: str) -> str:
    filters = []
    if oa:
        filters.append("open_access.is_oa:true")
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    params = {
        "search": query,
        "per-page": str(min(max(limit, 1), 200)),
        "select": SELECT_FIELDS,
    }
    if filters:
        params["filter"] = ",".join(filters)
    if mailto:
        params["mailto"] = mailto
    return API + "?" + urllib.parse.urlencode(params)


def http_get_json(url: str, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pdf_url_of(work: dict) -> str | None:
    loc = work.get("best_oa_location") or {}
    return loc.get("pdf_url")


def doi_of(work: dict) -> str:
    return (work.get("doi") or "").replace("https://doi.org/", "")


def short(text: str, n: int = 72) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def fetch_works(query: str, oa: bool, from_year: int, limit: int, mailto: str) -> list:
    url = build_url(query, oa, from_year, limit, mailto)
    data = http_get_json(url)
    return data.get("results", [])


def compact(work: dict) -> dict:
    return {
        "id": work.get("id"),
        "doi": doi_of(work),
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "is_oa": bool((work.get("open_access") or {}).get("is_oa")),
        "pdf_url": pdf_url_of(work),
    }


def print_table(works: list) -> None:
    if not works:
        print("(no results)")
        return
    for i, w in enumerate(works, 1):
        oa = "OA" if (w.get("open_access") or {}).get("is_oa") else "--"
        print(f"[{i:02d}] {w.get('publication_year')} | {oa} | cited {w.get('cited_by_count', 0)}")
        print(f"     {short(w.get('title'))}")
        d = doi_of(w)
        if d:
            print(f"     DOI: {d}")
        p = pdf_url_of(w)
        if p:
            print(f"     PDF: {p}")
        print(f"     {w.get('id')}")


def safe_filename(work: dict) -> str:
    year = work.get("publication_year") or "n.d."
    title = work.get("title") or "untitled"
    stem = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_")[:80]
    return f"{year}_{stem}.pdf"


def download(url: str, path: str, timeout: float = 180.0, attempts: int = 3) -> str | None:
    """带断点续传的下载。成功后返回最终路径，失败返回 None。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    part = path + ".part"
    for attempt in range(1, attempts + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": USER_AGENT}
        mode = "wb"
        if done:
            headers["Range"] = f"bytes={done}-"
            mode = "ab"
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=timeout)
            try:
                status = getattr(resp, "status", 200)
                if done and status == 200:
                    mode = "wb"  # 服务器不支持断点续传，重写
                with open(part, mode) as f:
                    shutil.copyfileobj(resp, f, length=1 << 16)
            finally:
                resp.close()
            os.replace(part, path)
            return path
        except Exception as e:  # noqa: BLE001
            print(f"    [download] 第 {attempt}/{attempts} 次失败: {e}", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAlex 文献检索（MCPC 共享文献区）")
    parser.add_argument("query", help="检索词")
    parser.add_argument("--oa", action="store_true", help="仅开放获取")
    parser.add_argument("--from-year", type=int, default=None, help="起始发表年份")
    parser.add_argument("--limit", type=int, default=10, help="返回条数（默认 10，最大 200）")
    parser.add_argument("--mailto", default=DEFAULT_MAILTO, help="礼貌池联系邮箱")
    parser.add_argument("--json", action="store_true", dest="json_out", help="JSON 输出")
    parser.add_argument("--download-dir", default=None, help="下载 OA PDF 到该目录")
    parser.add_argument("--download-limit", type=int, default=0, help="最多下载几篇（默认 0，不下载）")
    args = parser.parse_args()

    try:
        works = fetch_works(args.query, args.oa, args.from_year, args.limit, args.mailto)
    except Exception as e:  # noqa: BLE001
        print(f"检索失败: {e}", file=sys.stderr)
        return 1

    if args.json_out:
        print(json.dumps([compact(w) for w in works], ensure_ascii=False, indent=2))
    else:
        print(f"OpenAlex 命中 {len(works)} 条（query: {args.query!r}）")
        print("-" * 72)
        print_table(works)

    if args.download_dir and args.download_limit > 0:
        os.makedirs(args.download_dir, exist_ok=True)
        n_done = 0
        for w in works:
            if n_done >= args.download_limit:
                break
            p = pdf_url_of(w)
            if not p:
                continue
            target = os.path.join(args.download_dir, safe_filename(w))
            if os.path.exists(target):
                print(f"[skip] 已存在: {target}")
                n_done += 1
                continue
            print(f"[download] {w.get('title')}")
            if download(p, target) is not None:
                print(f"    -> {target}")
                n_done += 1
            else:
                print(f"    -> 下载失败（稍后可重试，支持断点续传）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
