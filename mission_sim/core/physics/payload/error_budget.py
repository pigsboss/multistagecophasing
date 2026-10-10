# mission_sim/core/physics/payload/error_budget.py
"""
MCPC L3: 级间误差预算工具
--------------------------
按 DD-001 R-L3-04：对各项误差分配/实测做自动合成校验（RSS 合成），
输出"预算闭合"与"逐项超分配"两类结论。
"""

import numpy as np


class ErrorBudget:
    """误差预算：总预算 + 各项分配（可选实测值），RSS 合成校验。"""

    def __init__(self, total_budget_m: float, name: str = ""):
        self.total_budget_m = float(total_budget_m)
        self.name = name
        self._items = []

    def add_item(self, name: str, allocation_m: float, measured_m: float = None) -> None:
        self._items.append({
            "name": name,
            "allocation_m": float(allocation_m),
            "measured_m": None if measured_m is None else float(measured_m),
        })

    def check(self) -> dict:
        """返回：closure_ok（RSS 合成 ≤ 总预算）、all_items_ok（实测 ≤ 分配）、violations 等。"""
        alloc = np.array([it["allocation_m"] for it in self._items], dtype=np.float64)
        rss = float(np.sqrt(np.sum(alloc ** 2))) if alloc.size else 0.0
        violations = [
            it["name"] for it in self._items
            if it["measured_m"] is not None and it["measured_m"] > it["allocation_m"]
        ]
        return {
            "name": self.name,
            "total_budget_m": self.total_budget_m,
            "rss_allocation_m": rss,
            "closure_ok": bool(rss <= self.total_budget_m + 1e-30),
            "all_items_ok": len(violations) == 0,
            "violations": violations,
            "n_items": len(self._items),
        }

    def __repr__(self) -> str:
        return (f"ErrorBudget({self.name or '-'}: total={self.total_budget_m:.2e} m, "
                f"items={len(self._items)})")
