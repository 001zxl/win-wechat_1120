"""人工接管状态（持久化）：接管期间禁止该会话的自动回复，直到显式恢复。"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List


class HandoffStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._items: Dict[str, Dict] = {}
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self._items = {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            self._items = {}

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self._items, handle, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def is_taken(self, peer: str) -> bool:
        with self._lock:
            return peer in self._items

    def take(self, peer: str, reason: str = "", by: str = "auto") -> None:
        if not peer:
            return
        with self._lock:
            if peer not in self._items:
                self._items[peer] = {"since": time.time(), "reason": reason, "by": by}
                self._save()

    def release(self, peer: str) -> bool:
        with self._lock:
            existed = self._items.pop(peer, None) is not None
            if existed:
                self._save()
            return existed

    def items(self) -> List[Dict]:
        with self._lock:
            return [{"peer": peer, **value} for peer, value in self._items.items()]
