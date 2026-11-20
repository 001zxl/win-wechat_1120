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
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    peer TEXT,
                    signature TEXT,
                    text TEXT,
                    state TEXT,
                    detail TEXT,
                    evidence TEXT,
                    event INTEGER DEFAULT 0
                )
                """
            )
            try:
                columns = [row[1] for row in
                           self._connection.execute("PRAGMA table_info(sends)").fetchall()]
                if "event" not in columns:
                    self._connection.execute("ALTER TABLE sends ADD COLUMN event INTEGER DEFAULT 0")
            except Exception:
                pass
            # 待重试队列：和发送台账同库，重启后仍能自动补回
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retries (
                    peer TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    signature TEXT,
                    event_seq INTEGER,
                    status TEXT,
                    attempts INTEGER,
                    next_at REAL,
                    last_error TEXT
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

    def record_send(self, peer: str, signature: str, text: str, state: str,
                    detail: str = "", evidence: str = "", event: int = 0) -> None:
        """发送台账：按「会话 + 消息内容 + 轮次」记账，同一事件不重复发，新事件照常处理。"""
        with self._lock:
            self._connection.execute(
                "INSERT INTO sends (ts, peer, signature, text, state, detail, evidence, event) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), peer, signature, text, state, detail, evidence, int(event)),
            )
            self._connection.commit()

    def max_event(self, peer: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(event), 0) FROM sends WHERE peer = ?", (peer,)
            ).fetchone()
        try:
            return int(row[0] or 0)
        except Exception:
            return 0

    # ---------- 待重试队列（持久化） ----------
    def upsert_retry(self, peer: str, signature: str, event_seq: int, status: str,
                     attempts: int, next_at: float, last_error: str = "") -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO retries (peer, ts, signature, event_seq, status, attempts, next_at, last_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(peer) DO UPDATE SET ts=excluded.ts, signature=excluded.signature, "
                "event_seq=excluded.event_seq, status=excluded.status, attempts=excluded.attempts, "
                "next_at=excluded.next_at, last_error=excluded.last_error",
                (peer, time.time(), signature, int(event_seq), status, int(attempts),
                 float(next_at), last_error),
            )
            self._connection.commit()

    def delete_retry(self, peer: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM retries WHERE peer = ?", (peer,))
            self._connection.commit()

    def list_retries(self):
        with self._lock:
            return self._connection.execute(
                "SELECT peer, signature, event_seq, status, attempts, next_at, last_error "
                "FROM retries ORDER BY next_at ASC"
            ).fetchall()

    def find_send(self, peer: str, signature: str):
        with self._lock:
            return self._connection.execute(
                "SELECT ts, peer, signature, text, state, detail FROM sends "
                "WHERE peer = ? AND signature = ? ORDER BY id DESC LIMIT 1",
                (peer, signature),
            ).fetchone()

    def recent_sends(self, limit: int = 20):
        with self._lock:
            return self._connection.execute(
                "SELECT ts, peer, text, state, detail FROM sends ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

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
