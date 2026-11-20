"""发送风控：拟人节奏、频次上限、活跃时段，降低自动化特征。"""
from __future__ import annotations

import random
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple


class SendRiskController:
    def __init__(self, safety, now=time.monotonic) -> None:
        self.enabled = bool(getattr(safety, "risk_enabled", True))
        self.min_interval = float(getattr(safety, "risk_min_interval_s", 20))
        self.max_per_hour = int(getattr(safety, "risk_max_per_hour", 10))
        self.max_per_day = int(getattr(safety, "risk_max_per_day", 50))
        self.chars_per_second = float(getattr(safety, "typing_chars_per_second", 4.0))
        self.jitter = float(getattr(safety, "risk_jitter", 0.3))
        self.max_wait = float(getattr(safety, "risk_max_wait_s", 15))
        self._now = now
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}
        self._recent: Deque[float] = deque()
        self._day_key = ""
        self._day_count = 0

    def plan(self, peer: str, text: str) -> Tuple[Optional[float], str]:
        """返回 (需要等待秒数, 原因)；None 表示触发上限，应拒绝发送。"""
        if not self.enabled:
            return 0.0, ""
        now = self._now()
        with self._lock:
            day = time.strftime("%Y-%m-%d")
            if day != self._day_key:
                self._day_key, self._day_count = day, 0
            if self._day_count >= self.max_per_day:
                return None, f"达到每日发送上限 {self.max_per_day}"
            while self._recent and now - self._recent[0] > 3600:
                self._recent.popleft()
            if len(self._recent) >= self.max_per_hour:
                return None, f"达到每小时发送上限 {self.max_per_hour}"
            since_last = now - self._last.get(peer, 0.0)
            cooldown = max(0.0, self.min_interval - since_last)
        typing = max(0.6, len(text) / max(0.5, self.chars_per_second))
        typing *= 1.0 + random.uniform(-self.jitter, self.jitter)
        wait = min(self.max_wait, max(cooldown, typing))
        return round(wait, 2), ("拟人输入节奏" if typing > cooldown else "会话冷却")

    def record(self, peer: str) -> None:
        now = self._now()
        with self._lock:
            self._last[peer] = now
            self._recent.append(now)
            day = time.strftime("%Y-%m-%d")
            if day != self._day_key:
                self._day_key, self._day_count = day, 0
            self._day_count += 1
