from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

_EMOJI_PATTERN = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\ufe0f]"
)
_SPACES = re.compile(r"\s+")
_GROUP_SIZE = re.compile(r"[\(（]\s*\d+\s*[\)）]")
_GROUP_WORDS = ("群聊", "工作群", "交流群", "互助群", "通知群", "家族群", "项目群",
                "事业部", "项目组", "客服中心")
_SERVICE_ACCOUNTS = ("微信团队", "订阅号", "服务号", "公众号", "微信支付",
                     "腾讯新闻", "文件传输助手", "微信运动", "微信收款助手")


def normalize_name(text: Optional[str]) -> str:
    if not text:
        return ""
    value = _EMOJI_PATTERN.sub("", str(text)).strip()
    value = _SPACES.sub("", value)
    value = value.replace("：", "").replace(":", "")
    value = value.lower()
    # OCR 可能把未读数字并进名字（“3张三”）
    value = re.sub(r"^\d{1,2}(?=[^\d])", "", value)
    return value


def is_group_name(text: Optional[str]) -> bool:
    if not text:
        return False
    raw = str(text)
    if _GROUP_SIZE.search(raw):
        return True
    return any(word in raw for word in _GROUP_WORDS)


def is_name_matched_strict(ocr_text: Optional[str], names: List[str]) -> Tuple[bool, Optional[str]]:
    """严格白名单匹配，防止名字前缀误命中。

    1) 归一化后全等
    2) OCR 比白名单多出的字符 <= 3（防“张三”命中“张三丰大碗面”）
    3) 仅允许 OCR 比白名单少 1 个字符（尾部轻微截断容错）
    """
    normalized = normalize_name(ocr_text)
    if not normalized:
        return False, None
    for name in names:
        target = normalize_name(name)
        if not target:
            continue
        if normalized == target:
            return True, name
        if normalized.startswith(target):
            tail = normalized[len(target):]
            # 名字后紧跟字母/数字/汉字 → 疑似更长的另一个名字（张三 vs 张三丰），拒绝
            if tail and re.match(r"[0-9a-z\u4e00-\u9fff]", tail[0]):
                continue
            return True, name
        if target.startswith(normalized) and len(target) - len(normalized) == 1:
            return True, name
    return False, None


def is_service_account(text: Optional[str]) -> bool:
    normalized = normalize_name(text)
    if not normalized:
        return False
    return any(normalize_name(item) in normalized for item in _SERVICE_ACCOUNTS)


class Whitelist:
    def __init__(self, names: List[str], allow_group_chat: bool = False) -> None:
        self.names = [name for name in (names or []) if name.strip()]
        self.allow_group_chat = allow_group_chat

    def matches(self, peer_text: Optional[str]) -> Tuple[bool, Optional[str]]:
        if not normalize_name(peer_text):
            return False, None
        if is_group_name(peer_text) and not self.allow_group_chat:
            return False, None
        matched, name = is_name_matched_strict(peer_text, self.names)
        if matched:
            return True, name
        # 系统服务号/公众号一律不自动回复（即使名字碰巧相似）
        if is_service_account(peer_text):
            return False, None
        return False, None


@dataclass
class Decision:
    action: str          # reply | escalate | block
    reason: str = ""
    text: str = ""


class Guardrails:
    def __init__(
        self,
        forbidden_keywords: List[str],
        escalate_keywords: List[str],
        forbidden_self_disclosure: List[str],
        max_reply_chars: int,
    ) -> None:
        self.forbidden = [k for k in forbidden_keywords if k]
        self.escalate = [k for k in escalate_keywords if k]
        self.disclosure = [k for k in forbidden_self_disclosure if k]
        self.max_chars = max_reply_chars

    def assess_incoming(self, incoming_text: str) -> Decision:
        text = incoming_text or ""
        for keyword in self.escalate:
            if keyword in text:
                return Decision("escalate", f"命中转人工关键词：{keyword}")
        return Decision("reply")

    def check_reply(self, reply: str) -> Decision:
        text = (reply or "").strip()
        if not text:
            return Decision("block", "回复为空")
        for keyword in self.forbidden:
            if keyword in text:
                return Decision("block", f"回复包含禁用词：{keyword}")
        for keyword in self.disclosure:
            if keyword in text:
                return Decision("block", f"回复暴露 AI 身份：{keyword}")
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip("，。,.!?！？ ")
            return Decision("reply", "已截断到长度上限", text)
        return Decision("reply", "", text)


class RateLimiter:
    def __init__(self, max_actions_per_minute: int, cooldown_s: float) -> None:
        self.max_actions = max_actions_per_minute
        self.cooldown_s = cooldown_s
        self._events: Deque[float] = deque()
        self._last_reply: Dict[str, float] = {}

    def _now(self) -> float:
        return time.monotonic()

    def allowed(self, peer_key: str) -> bool:
        now = self._now()
        while self._events and now - self._events[0] > 60:
            self._events.popleft()
        if len(self._events) >= self.max_actions:
            return False
        last = self._last_reply.get(peer_key)
        if last is None:  # 首次对话不做冷却
            return True
        return now - last >= self.cooldown_s

    def record(self, peer_key: str, cost: int = 1) -> None:
        now = self._now()
        for _ in range(max(1, cost)):
            self._events.append(now)
        self._last_reply[peer_key] = now

    def wait_seconds(self, peer_key: str) -> float:
        now = self._now()
        last = self._last_reply.get(peer_key)
        cooldown_wait = 0.0 if last is None else max(0.0, self.cooldown_s - (now - last))
        while self._events and now - self._events[0] > 60:
            self._events.popleft()
        window_wait = 0.0
        if len(self._events) >= self.max_actions and self._events:
            window_wait = max(0.0, 60 - (now - self._events[0]))
        return max(cooldown_wait, window_wait)


def in_quiet_hours(ranges: List[str], now: Optional[time.struct_time] = None) -> bool:
    current = now or time.localtime()
    minutes = current.tm_hour * 60 + current.tm_min
    for item in ranges or []:
        try:
            start_text, end_text = item.split("-")
            sh, sm = [int(x) for x in start_text.strip().split(":")]
            eh, em = [int(x) for x in end_text.strip().split(":")]
        except Exception:
            continue
        start, end = sh * 60 + sm, eh * 60 + em
        if start <= end:
            if start <= minutes <= end:
                return True
        else:  # 跨天，例如 23:00-08:00
            if minutes >= start or minutes <= end:
                return True
    return False


def group_allows(
    peer: str,
    items,
    allow_group_chat: bool,
    group_whitelist: List[str],
    mode: str = "mention",
    keywords: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """群聊回复策略：白名单 + @/关键词 门槛。"""
    if not is_group_name(peer):
        return True, ""
    if not allow_group_chat:
        return False, "群聊自动回复未开启"
    if group_whitelist:
        normalized = normalize_name(peer)
        if not any(normalize_name(name) in normalized or normalized in normalize_name(name)
                   for name in group_whitelist if name.strip()):
            return False, "群不在群白名单"
    texts = []
    mention = False
    for item in items or []:
        texts.append(getattr(item, "text", "") or "")
        if getattr(item, "mention", False):
            mention = True
    joined = " ".join(texts)
    if mode == "all":
        return True, ""
    if mode == "keyword":
        words = [word for word in (keywords or []) if word]
        if any(word in joined for word in words):
            return True, ""
        return False, "群消息未命中关键词"
    # 默认 mention
    if mention or "@所有人" in joined or "@我" in joined:
        return True, ""
    return False, "群里没有 @我"
