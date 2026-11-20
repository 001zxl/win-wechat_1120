from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple


class MemoryStore:
    """长程会话记忆：最近消息 + 滚动摘要 + 客户档案。"""

    def __init__(self, db_path: str, recent_limit: int = 8, summarize_after: int = 12) -> None:
        self.db_path = db_path
        self.recent_limit = recent_limit
        self.summarize_after = summarize_after
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    peer TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    peer TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    note TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    summarized_id INTEGER DEFAULT 0,
                    updated_at REAL DEFAULT 0
                )
                """
            )
            self._db.commit()

    # ---------- 消息 ----------
    def add(self, peer: str, direction: str, kind: str, text: str) -> None:
        if not peer or not text:
            return
        with self._lock:
            self._db.execute(
                "INSERT INTO messages (peer, direction, kind, text, ts) VALUES (?, ?, ?, ?, ?)",
                (peer, direction, kind, text, time.time()),
            )
            self._db.commit()

    def recent(self, peer: str, limit: Optional[int] = None, upto_id: Optional[int] = None) -> List[Tuple[int, str, str, str]]:
        limit = limit or self.recent_limit
        with self._lock:
            if upto_id is not None:
                rows = self._db.execute(
                    "SELECT id, direction, kind, text FROM messages WHERE peer=? AND id<? ORDER BY id DESC LIMIT ?",
                    (peer, upto_id, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT id, direction, kind, text FROM messages WHERE peer=? ORDER BY id DESC LIMIT ?",
                    (peer, limit),
                ).fetchall()
        return list(reversed(rows))

    def history_lines(self, peer: str, limit: Optional[int] = None) -> List[str]:
        lines = []
        for _id, direction, kind, text in self.recent(peer, limit):
            prefix = "对方" if direction == "in" else "我方"
            suffix = "" if kind == "text" else f"[{kind}]"
            lines.append(f"{prefix}{suffix}：{text}")
        return lines

    def last_id(self, peer: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT MAX(id) FROM messages WHERE peer=?", (peer,)).fetchone()
        return int(row[0] or 0)

    # ---------- 档案/摘要 ----------
    def profile(self, peer: str) -> Dict[str, str]:
        with self._lock:
            row = self._db.execute(
                "SELECT name, note, summary, summarized_id FROM profiles WHERE peer=?", (peer,)
            ).fetchone()
        if not row:
            return {"name": peer, "note": "", "summary": "", "summarized_id": 0}
        return {"name": row[0] or peer, "note": row[1] or "", "summary": row[2] or "",
                "summarized_id": int(row[3] or 0)}

    def set_profile(self, peer: str, name: str = "", note: str = "") -> None:
        current = self.profile(peer)
        with self._lock:
            self._db.execute(
                """
                INSERT INTO profiles (peer, name, note, summary, summarized_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(peer) DO UPDATE SET name=excluded.name, note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (peer, name or current["name"], note or current["note"],
                 current["summary"], current["summarized_id"], time.time()),
            )
            self._db.commit()

    def set_summary(self, peer: str, summary: str, summarized_id: int) -> None:
        current = self.profile(peer)
        with self._lock:
            self._db.execute(
                """
                INSERT INTO profiles (peer, name, note, summary, summarized_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(peer) DO UPDATE SET summary=excluded.summary,
                    summarized_id=excluded.summarized_id, updated_at=excluded.updated_at
                """,
                (peer, current["name"], current["note"], summary, summarized_id, time.time()),
            )
            self._db.commit()

    def needs_summary(self, peer: str) -> bool:
        profile = self.profile(peer)
        return (self.last_id(peer) - profile["summarized_id"]) >= self.summarize_after

    def pending_messages(self, peer: str) -> List[Tuple[int, str, str, str]]:
        profile = self.profile(peer)
        with self._lock:
            rows = self._db.execute(
                "SELECT id, direction, kind, text FROM messages WHERE peer=? AND id>? ORDER BY id",
                (peer, profile["summarized_id"]),
            ).fetchall()
        return list(rows)

    def close(self) -> None:
        try:
            with self._lock:
                self._db.close()
        except Exception:
            pass
