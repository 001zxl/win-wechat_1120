"""事实依据闸门：判断"能不能回答"而不是"能不能生成"。

设计原则（对应复查 1/2 项）：

1. 只拦**事实性承诺**：价格/库存/配送/售后/规格/账期/交期/认证/公司信息。
   寒暄、确认收到、闲聊不拦。
2. 合法依据只有两种：结构化工具结果（ok=True）和知识库命中片段
   （片段里必须出现该类别的依据词，不能"随便返回一段"就算有依据）。
3. 缺依据时：能把缺失字段问清楚的（型号/数量）先追问，追问不解决就转人工；
   不生成任何带数字/承诺的回复。
4. 生成后再做一次"数字溯源"：回复里的数字+单位必须能在客户原话或依据里找到，
   防止模型自己改价、改交期、改库存。
5. 每次客户消息变化后都要重新判定（`decide` 是纯函数，agent 在每轮循环里重复调用）。

这里不引入 NLP 依赖：只用关键词 + 数字溯源，宁可多转人工，也不让模型自由发挥。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# 类别词表
#   ask   = 客户在问什么（触发判定）
#   proof = 依据里必须出现什么（证明这条依据确实在讲这件事）
# ---------------------------------------------------------------------------
RULES: Dict[str, Dict[str, Sequence[str]]] = {
    "price": {
        "ask": ("价格", "什么价", "怎么卖", "多少钱", "报价", "单价", "美金", "美元",
                "折扣", "优惠", "便宜", "降价", "出厂价", "批发价", "含税", "fob", "price",
                "quote", "cost", "usd", "moq", "起订"),
        "proof": ("价格", "价", "售价", "报价", "单价", "美金", "美元", "元", "折扣",
                  "含税", "fob", "usd", "price", "quote", "cost", "起订", "moq"),
    },
    "stock": {
        "ask": ("库存", "有货", "现货", "缺货", "备货", "还有吗", "有没有货", "多久能有",
                "stock", "inventory"),
        "proof": ("库存", "现货", "有货", "缺货", "备货", "stock", "inventory", "件"),
    },
    "shipping": {
        "ask": ("发货", "物流", "快递", "配送", "送到", "运费", "到货", "寄到", "能到",
                "偏远", "西藏", "新疆", "内蒙古", "海南", "香港", "澳门", "台湾",
                "国外", "出口", "几天到", "多久到", "包邮", "能不能发",
                "shipping", "delivery", "freight"),
        "proof": ("发货", "物流", "快递", "配送", "运费", "到货", "运输", "顺丰", "德邦",
                  "偏远", "西藏", "新疆", "内蒙古", "海南", "港澳", "台湾", "出口",
                  "报关", "包邮", "空运", "海运", "陆运", "shipping", "delivery"),
    },
    "after_sales": {
        "ask": ("售后", "退货", "换货", "退款", "保修", "质保", "维修", "坏了", "破损",
                "漏发", "少发", "补发", "发票", "三包", "退换", "refund", "warranty"),
        "proof": ("售后", "退货", "换货", "退款", "保修", "质保", "维修", "破损", "漏发",
                  "少发", "补发", "发票", "三包", "退换", "refund", "warranty", "return"),
    },
    "spec": {
        "ask": ("规格", "尺寸", "材质", "参数", "重量", "颜色", "容量", "功率", "型号",
                "直径", "厚度", "长度", "宽度", "高度", "净重", "毛重", "包装",
                "spec", "specification"),
        "proof": ("规格", "尺寸", "材质", "参数", "重量", "颜色", "容量", "功率", "型号",
                  "直径", "厚度", "长度", "宽度", "高度", "净重", "毛重", "包装",
                  "mm", "cm", "kg", "spec", "specification"),
    },
    "payment": {
        "ask": ("付款", "支付", "定金", "预付", "账期", "开票", "合同", "信用证",
                "电汇", "支付宝", "微信支付", "tt", "lc", "payment", "deposit"),
        "proof": ("付款", "支付", "定金", "预付", "账期", "发票", "合同", "信用证",
                  "电汇", "支付宝", "微信", "tt", "lc", "payment", "deposit", "t/t"),
    },
    "lead_time": {
        "ask": ("交期", "工期", "货期", "多久能", "什么时候发", "几天能", "生产周期",
                "lead time", "leadtime"),
        "proof": ("交期", "工期", "货期", "生产周期", "发货时间", "工作日", "天内",
                  "天左右", "天发货", "lead time", "leadtime"),
    },
    "cert": {
        "ask": ("认证", "证书", "检测报告", "资质", "ce", "rohs", "fda", "iso", "3c",
                "certification", "certificate"),
        "proof": ("认证", "证书", "检测报告", "资质", "质检", "ce", "rohs", "fda",
                  "iso", "3c", "certification", "certificate"),
    },
    "company": {
        "ask": ("工厂", "地址", "在哪", "公司", "规模", "成立", "厂房", "参观", "参观工厂"),
        "proof": ("工厂", "地址", "位于", "公司", "成立", "规模", "厂房", "车间",
                  "工业区", "园区"),
    },
    "custom": {
        "ask": ("定制", "打样", "样品", "logo", "丝印", "贴牌", "改颜色", "加印", "开模"),
        "proof": ("定制", "打样", "样品", "logo", "丝印", "贴牌", "起订", "开模", "打样费"),
    },
}

CATEGORY_LABELS = {
    "price": "价格", "stock": "库存", "shipping": "配送/物流", "after_sales": "售后",
    "spec": "规格/参数", "payment": "付款/账期", "lead_time": "交期", "cert": "认证",
    "company": "公司/工厂信息", "custom": "定制/打样",
}

# 型号：FT-120 / fc_200 / SKU-XYZ 都算"客户已经点名了型号"
_SKU_RE = re.compile(r"[A-Za-z]{2,8}[-_ ]?\d{2,7}|[A-Z]{2,8}-[A-Z]{2,8}")
# 指代：出现这些词且没有型号时，允许从上下文补型号
_PRONOUNS = ("这个", "这款", "该产品", "该款", "那个", "那款", "它", "此款", "这个产品",
             "同样的", "同一款", "上述")

# 回复里的数字承诺：数字 + 单位
_CLAIM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(美金|美元|元|块钱|块|件|个|pcs|套|天|周|小时|公斤|kg|克|g|"
    r"毫米|mm|厘米|cm|米|%|折)",
    re.I,
)


def _contains(text: str, token: str) -> bool:
    """ASCII 词按词边界匹配（防止 ce 命中 price），中文直接子串。"""
    if not text or not token:
        return False
    lowered = text.lower()
    token = token.lower()
    if token.isascii():
        return re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", lowered) is not None
    return token in lowered


def _matches_any(text: str, tokens: Iterable[str]) -> Optional[str]:
    for token in tokens:
        if _contains(text, token):
            return token
    return None


def detect_categories(text: str) -> List[str]:
    """客户这句话在问哪些事实类别（按 RULES 顺序，保证结果稳定）。"""
    found: List[str] = []
    for category, rule in RULES.items():
        if _matches_any(text or "", rule["ask"]):
            found.append(category)
    return found


def covers(category: str, text: str) -> bool:
    rule = RULES.get(category)
    if not rule or not text:
        return False
    return _matches_any(text, rule["proof"]) is not None


def has_sku(text: str) -> bool:
    return _SKU_RE.search(text or "") is not None


def has_pronoun(text: str) -> bool:
    return any(word in (text or "") for word in _PRONOUNS)


def skus_in(text: str) -> List[str]:
    return re.findall(_SKU_RE, text or "")


# ---------------------------------------------------------------------------
# 依据
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    """一次判定用到的全部依据，同时也是写进草稿/审计的"依据快照"。"""

    tool_results: List[Any] = field(default_factory=list)
    chunks: List[Any] = field(default_factory=list)
    provider: str = "local"
    kb_version: str = ""
    kb_error: str = ""

    # ---- 文本 ----
    def tool_text(self) -> str:
        lines = []
        for result in self.tool_results:
            ok = bool(getattr(result, "ok", False))
            if not ok:
                continue
            lines.append(str(getattr(result, "summary", "")))
            data = getattr(result, "data", None)
            if isinstance(data, dict):
                lines.append(" ".join(str(value) for value in data.values()))
        return "\n".join(line for line in lines if line)

    def kb_text(self) -> str:
        parts = []
        for chunk in self.chunks:
            parts.append(str(getattr(chunk, "text", chunk)))
            source = str(getattr(chunk, "source", "") or "")
            if source:
                parts.append(source)
            meta = getattr(chunk, "meta", None)
            if isinstance(meta, dict):
                parts.append(" ".join(str(value) for value in meta.values()))
        return "\n".join(part for part in parts if part)

    def text(self) -> str:
        return "\n".join(part for part in (self.tool_text(), self.kb_text()) if part)

    # ---- 覆盖情况 ----
    def tool_categories(self) -> Set[str]:
        covered: Set[str] = set()
        for result in self.tool_results:
            if not bool(getattr(result, "ok", False)):
                continue
            name = str(getattr(result, "name", ""))
            if name == "price_lookup":
                covered.add("price")
            elif name == "stock_check":
                covered.add("stock")
            summary = str(getattr(result, "summary", ""))
            for category in RULES:
                if covers(category, summary):
                    covered.add(category)
        return covered

    def kb_categories(self) -> Set[str]:
        text = self.kb_text()
        if not text:
            return set()
        return {category for category in RULES if covers(category, text)}

    def covered(self) -> Set[str]:
        return self.tool_categories() | self.kb_categories()

    # ---- 序列化（草稿 / 审计 / 控制台） ----
    def sources(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for chunk in self.chunks:
            rows.append({
                "source": str(getattr(chunk, "source", "") or ""),
                "score": round(float(getattr(chunk, "score", 0.0) or 0.0), 4),
                "id": str(getattr(chunk, "chunk_id", "") or ""),
                "title": str((getattr(chunk, "meta", {}) or {}).get("title", "") or ""),
            })
        return rows

    def summary(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "kb_version": self.kb_version,
            "kb_error": self.kb_error,
            "tool": [str(getattr(result, "name", "")) for result in self.tool_results
                     if bool(getattr(result, "ok", False))],
            "sources": self.sources(),
        }


@dataclass
class Verdict:
    action: str                     # ok | ask | handoff
    categories: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    reason: str = ""
    reply: str = ""                 # action=ask：确定性追问（不含任何事实承诺）
    ask: str = ""                   # product | quantity | ambiguous

    @property
    def ok(self) -> bool:
        return self.action == "ok"


def _ask_for_missing(missing: Sequence[str], evidence: Evidence,
                     product_hint: str = "") -> Optional[Verdict]:
    """能用一句追问解决就追问，解决不了返回 None（调用方转人工）。"""
    if set(missing) - {"price", "stock"}:
        return None
    name = ""
    need = ""
    for result in evidence.tool_results:
        if bool(getattr(result, "ok", False)):
            continue
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        need = str(data.get("need") or need)
        name = str(data.get("name") or data.get("sku") or name)
    if not need and product_hint:
        need = "quantity"
    if need == "quantity":
        label = name or product_hint or "这款产品"
        return Verdict("ask", list(missing), [], "缺少数量，先追问", 
                       f"{label} 的价格要看数量，您大概需要多少件？", "quantity")
    if need == "product" and not has_sku(product_hint):
        return Verdict("ask", list(missing), [], "缺少型号，先追问",
                       "请问您要哪一款产品？方便发一下型号吗？", "product")
    return None


def decide(text: str, evidence: Evidence, product_hint: str = "",
           ambiguous_product: str = "") -> Verdict:
    """判断这句话能不能自动回答。

    - action=ok       有适用依据（或这句话本来就不涉及事实承诺）
    - action=ask      缺的字段能问清楚，用固定话术追问
    - action=handoff  缺依据且问不清楚 → 转人工
    """
    if ambiguous_product:
        return Verdict("ask", [], [], "型号不唯一，先追问", ambiguous_product, "ambiguous")
    categories = detect_categories(text)
    if not categories:
        return Verdict("ok")
    covered = evidence.covered()
    missing = [category for category in categories if category not in covered]
    if not missing:
        return Verdict("ok", categories)
    asked = _ask_for_missing(missing, evidence, product_hint)
    if asked is not None:
        asked.categories = categories
        return asked
    labels = "、".join(CATEGORY_LABELS.get(item, item) for item in missing)
    return Verdict("handoff", categories, missing, f"缺少可用依据：{labels}")


def find_unsupported_claim(reply: str, evidence_text: str, incoming_text: str) -> str:
    """回复里的"数字+单位"必须能溯源到客户原话或依据，否则返回那个片段。"""
    allowed = f"{incoming_text or ''}\n{evidence_text or ''}"
    for match in _CLAIM_RE.finditer(reply or ""):
        number = match.group(1)
        if number in allowed:
            continue
        # 允许 "3-5 天" 这种区间：只要区间两端都能溯源就不算编造
        return match.group(0).strip()
    return ""


def summarize_for_prompt(verdict: Verdict) -> str:
    """给模型的硬约束（只有 action=ok 时才会走到生成，这里兜底再强调一次）。"""
    if verdict.ok:
        return ""
    labels = "、".join(CATEGORY_LABELS.get(item, item) for item in verdict.missing)
    return f"本次问题涉及{labels}，没有可用依据：不要给具体数字或承诺，只能追问或说明需要确认。"
