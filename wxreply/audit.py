from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Optional

from PIL import Image


class AuditStore:
    """SQLite 审计；连接跨线程共享，写入加锁。"""

    def __init__(self, db_path: str, evidence_dir: str, save_screenshots: bool = True,
                 skip_throttle_s: float = 300.0) -> None:
        self.db_path = db_path
        self.evidence_dir = evidence_dir
        self.save_screenshots = save_screenshots
        # 同一会话、同一跳过原因在窗口期内只落库一次：扫描每秒一轮，否则审计库会被"跳过"撑爆
        self.skip_throttle_s = skip_throttle_s
        self._skip_seen: dict = {}
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        if save_screenshots:
            os.makedirs(evidence_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    peer TEXT,
                    action TEXT,
                    incoming TEXT,
                    reply TEXT,
                    status TEXT,
                    detail TEXT,
                    model TEXT,
                    latency_ms INTEGER,
                    evidence_path TEXT
                )
                """
            )
            self._connection.commit()

    def record(
        self,
        peer: str,
        action: str,
        incoming: str = "",
        reply: str = "",
        status: str = "",
        detail: str = "",
        model: str = "",
        latency_ms: int = 0,
        image: Optional[Image.Image] = None,
    ) -> None:
        now = time.time()
        if action == "skip":
            key = (peer, status, detail)
            with self._lock:
                last = self._skip_seen.get(key)
                if last is not None and now - last < self.skip_throttle_s:
                    return                      # 窗口期内重复跳过：不写库
                self._skip_seen[key] = now
                if len(self._skip_seen) > 1024:
                    cutoff = now - self.skip_throttle_s
                    self._skip_seen = {k: v for k, v in self._skip_seen.items() if v >= cutoff}
        evidence_path = ""
        if image is not None and self.save_screenshots:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{stamp}-{abs(hash(peer)) % 10**6:06d}.png"
            evidence_path = os.path.join(self.evidence_dir, filename)
            try:
                image.save(evidence_path)
            except Exception:
                evidence_path = ""
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO events (ts, peer, action, incoming, reply, status, detail, model, latency_ms, evidence_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (time.time(), peer, action, incoming, reply, status, detail, model, latency_ms, evidence_path),
            )
            self._connection.commit()

    def recent(self, limit: int = 20):
        with self._lock:
            return self._connection.execute(
                "SELECT ts, peer, action, status, reply, detail FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def close(self) -> None:
        try:
            with self._lock:
                self._connection.close()
        except Exception:
            pass
