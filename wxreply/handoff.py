"""人工接管状态（持久化）：接管期间禁止该会话的自动回复，直到显式恢复。"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from typing import Dict, List


class HandoffStore:
    """原子保存 + 备份恢复；文件损坏时不再静默当作空名单。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.backup_path = path + ".bak"
        self.corrupt = False
        self.recovered_from = ""
        self.load_error = ""
        self._lock = threading.Lock()
        self._items: Dict[str, Dict] = {}
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("handoff 文件不是对象")
            self._items = {str(k): v for k, v in data.items() if isinstance(v, dict)}
            return
        except Exception as exc:
            self.load_error = str(exc)
        # 读取失败：保留现场，尝试从备份恢复；恢复不了就把 corrupt 置位，禁止自动发送
        self.corrupt = True
        try:
            shutil.copyfile(self.path, self.path + ".corrupt")
        except Exception:
            pass
        if os.path.exists(self.backup_path):
            try:
                with open(self.backup_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self._items = {str(k): v for k, v in data.items() if isinstance(v, dict)}
                    self.recovered_from = self.backup_path
                    self.corrupt = False
            except Exception:
                pass

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._items, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)          # 原子替换
            try:
                shutil.copyfile(self.path, self.backup_path)
            except Exception:
                pass
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
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
