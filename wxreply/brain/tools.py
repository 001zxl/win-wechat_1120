from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolResult:
    name: str
    ok: bool
    summary: str
    data: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "summary": self.summary, "data": self.data}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


class ToolBox:
    """业务工具：查报价、查库存、建线索、转人工。数据来自本地 JSON。"""

    def __init__(self, price_file: str, leads_file: str, enabled: bool = True) -> None:
        self.price_file = price_file
        self.leads_file = leads_file
        self.enabled = enabled
        self.products: List[Dict[str, Any]] = []
        self.stock: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.price_file):
            return
        try:
            with open(self.price_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.products = data.get("products") or []
            self.stock = data.get("stock") or {}
        except Exception:
            self.products = []

    # ---------- 匹配 ----------
    def _match_product(self, text: str) -> Optional[Dict[str, Any]]:
        target = _norm(text)
        for product in self.products:
            names = [product.get("sku", ""), product.get("name", "")] + list(product.get("aliases") or [])
            for name in names:
                if name and _norm(name) in target:
                    return product
        if len(self.products) == 1:
            return self.products[0]
        return None

    @staticmethod
    def _parse_quantity(text: str) -> Optional[int]:
        # 必须带单位，避免把型号 FT-120 里的 120 当成数量
        match = re.search(r"(\d{2,7})\s*(?:件|个|pcs|套|台|条|只)", text or "", re.I)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    # ---------- 工具实现 ----------
    def price_lookup(self, product: str = "", quantity: Optional[int] = None) -> ToolResult:
        item = self._match_product(product)
        if not item:
            return ToolResult("price_lookup", False, "没有匹配到产品", {"product": product})
        ranges = item.get("price_ranges") or []
        matched = None
        qty = quantity
        if qty is not None:
            for band in ranges:
                if band.get("min_qty", 0) <= qty <= band.get("max_qty", 10**9):
                    matched = band
                    break
        selected = matched or (ranges[0] if ranges else None)
        if not selected:
            return ToolResult("price_lookup", False, "产品没有报价区间", {"sku": item.get("sku")})
        low, high = selected.get("price", [None, None])
        summary = f"{item.get('name') or item.get('sku')}：{selected.get('min_qty')}-{selected.get('max_qty')} 件，FOB {low}-{high} 美金"
        return ToolResult("price_lookup", True, summary, {
            "sku": item.get("sku"), "name": item.get("name"),
            "quantity": qty, "range": selected, "price": selected.get("price"),
        })

    def stock_check(self, product: str = "") -> ToolResult:
        item = self._match_product(product)
        if not item:
            return ToolResult("stock_check", False, "没有匹配到产品", {"product": product})
        sku = item.get("sku", "")
        count = self.stock.get(sku)
        if count is None:
            return ToolResult("stock_check", False, "没有库存数据", {"sku": sku})
        return ToolResult("stock_check", True, f"{item.get('name') or sku} 当前库存 {count} 件", {"sku": sku, "stock": count})

    def create_lead(self, company: str = "", note: str = "") -> ToolResult:
        record = {"company": company, "note": note, "ts": __import__("time").time()}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.leads_file)) or ".", exist_ok=True)
            with open(self.leads_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            return ToolResult("create_lead", False, f"写入线索失败：{exc}", record)
        return ToolResult("create_lead", True, f"已记录线索：{company}", record)

    def handoff(self, reason: str = "") -> ToolResult:
        return ToolResult("handoff", True, f"已标记转人工：{reason}", {"reason": reason})

    # ---------- 预取（不依赖 function calling，任何模型都能用） ----------
    def prefetch(self, text: str) -> List[ToolResult]:
        results: List[ToolResult] = []
        if not self.enabled:
            return results
        lowered = _norm(text)
        want_price = any(word in lowered for word in ("价", "报价", "多少钱", "价格", "美金", "usd"))
        want_stock = any(word in lowered for word in ("库存", "现货", "有货", "缺货"))
        if want_price or want_stock:
            product = self._match_product(text)
            quantity = self._parse_quantity(text)
            if want_price:
                results.append(self.price_lookup(text, quantity))
            if want_stock:
                results.append(self.stock_check(text))
            if product is None and not results:
                results.append(ToolResult("price_lookup", False, "没有匹配到产品", {}))
        return [item for item in results if item.ok]

    # ---------- function calling schema ----------
    def schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "price_lookup",
                    "description": "查询产品报价区间，参数 product 可以是产品名/SKU/客户原话，quantity 是数量",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "string", "description": "产品名或 SKU 或客户原话"},
                            "quantity": {"type": "integer", "description": "数量，未知可省略"},
                        },
                        "required": ["product"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "stock_check",
                    "description": "查询产品库存",
                    "parameters": {
                        "type": "object",
                        "properties": {"product": {"type": "string"}},
                        "required": ["product"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_lead",
                    "description": "记录一条销售线索",
                    "parameters": {
                        "type": "object",
                        "properties": {"company": {"type": "string"}, "note": {"type": "string"}},
                        "required": ["company"],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name == "price_lookup":
                result = self.price_lookup(arguments.get("product", ""), arguments.get("quantity"))
            elif name == "stock_check":
                result = self.stock_check(arguments.get("product", ""))
            elif name == "create_lead":
                result = self.create_lead(arguments.get("company", ""), arguments.get("note", ""))
            elif name == "handoff":
                result = self.handoff(arguments.get("reason", ""))
            else:
                return json.dumps({"ok": False, "error": f"unknown tool {name}"}, ensure_ascii=False)
            return json.dumps(result.as_dict(), ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
