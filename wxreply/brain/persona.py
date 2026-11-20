from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Persona:
    id: str = "default"
    display_name: str = "小助手"
    identity: Dict[str, Any] = field(default_factory=dict)
    language: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)
    boundaries: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, str]] = field(default_factory=list)
    scenario_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_chars(self) -> int:
        return int(self.style.get("max_chars", 120))

    def summary(self) -> str:
        identity = self.identity or {}
        role = identity.get("role", "")
        company = identity.get("company", "")
        region = identity.get("region", "")
        parts = [part for part in (company, role, region) if part]
        return "，".join(parts)

    def forbidden_topics(self) -> List[str]:
        return list(self.boundaries.get("forbidden_topics") or [])

    def must_escalate(self) -> List[str]:
        return list(self.boundaries.get("must_escalate") or [])


def load_persona(path: str) -> Persona:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Persona(
        id=str(data.get("id", "default")),
        display_name=str(data.get("display_name", "小助手")),
        identity=data.get("identity") or {},
        language=data.get("language") or {},
        style=data.get("style") or {},
        boundaries=data.get("boundaries") or {},
        examples=data.get("examples") or [],
        scenario_overrides=data.get("scenario_overrides") or {},
    )


def build_system_prompt(
    persona: Persona,
    kb_snippets: Optional[List[Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    style = persona.style or {}
    boundaries = persona.boundaries or {}
    lines = [
        f"你是{persona.display_name}，{persona.summary()}。",
        f"说话风格：{style.get('tone', '自然、专业')}；{style.get('typing_style', '口语化')}。",
        f"称呼习惯：{style.get('address_style', '称您')}。",
        f"回复长度：最多 {style.get('max_sentences', 3)} 句话 / {style.get('max_chars', 120)} 字。",
        f"表情使用：{style.get('emoji_policy', 'low')}。",
    ]
    if boundaries.get("price_policy"):
        lines.append(f"报价策略：{boundaries['price_policy']}。")
    if persona.forbidden_topics():
        lines.append("禁止话题：" + "、".join(persona.forbidden_topics()) + "。")
    if persona.must_escalate():
        lines.append("遇到以下情况必须建议转人工，不要自行承诺：" + "、".join(persona.must_escalate()) + "。")
    lines.append("绝对不要自称 AI、语言模型或机器人；不知道就说不确定，不要编造。")
    if profile:
        facts = []
        for key, label in (("name", "客户"), ("company", "公司"), ("note", "备注")):
            if profile.get(key):
                facts.append(f"{label}：{profile[key]}")
        if facts:
            lines.append("客户档案：" + "；".join(facts) + "。")
    if kb_snippets:
        lines.append("可参考的知识（不要照抄，用你的话回答）：")
        for index, chunk in enumerate(kb_snippets, 1):
            text = getattr(chunk, "text", str(chunk))
            source = getattr(chunk, "source", "")
            suffix = f"（来源：{source}）" if source else ""
            lines.append(f"[{index}] {text}{suffix}")
    return "\n".join(lines)


def build_messages(
    persona: Persona,
    incoming: List[str],
    kb_snippets: Optional[List[Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    history: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(persona, kb_snippets, profile)}
    ]
    for example in persona.examples or []:
        user = example.get("user")
        assistant = example.get("assistant")
        if user and assistant:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
    for item in history or []:
        messages.append({"role": "assistant", "content": item})
    user_text = "\n".join(text for text in incoming if text.strip())
    messages.append({"role": "user", "content": user_text or "（对方发来一条消息）"})
    return messages
