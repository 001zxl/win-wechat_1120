"""WeKnora 知识库适配器（可选）。

默认不启用：只有 rag.provider=weknora 且配置了 base_url 才会调用。
接口按"提交问题 → 返回带来源的片段"设计，部署 WeKnora 后若字段不同，
只需要改 parse_hits()，业务侧（agent）不用动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class KnowledgeHit:
    text: str
    source: str = ""
    score: float = 0.0


class WeKnoraClient:
    def __init__(self, base_url: str, api_key: str = "", search_path: str = "/api/v1/knowledge/search",
                 timeout: float = 8.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.search_path = search_path or "/api/v1/knowledge/search"
        self.timeout = timeout
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeHit]:
        if not self.enabled or not (query or "").strip():
            return []
        url = self.base_url + self.search_path
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"query": query, "top_k": int(top_k)}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            if response.status_code >= 400:
                self.last_error = f"HTTP {response.status_code}"
                return []
            data = response.json()
        except Exception as exc:
            self.last_error = str(exc)
            return []
        return self.parse_hits(data)

    @staticmethod
    def parse_hits(data: Any) -> List[KnowledgeHit]:
        """兼容几种常见返回：{data:[...]} / {results:[...]} / 直接数组。"""
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("data") or data.get("results") or data.get("hits") or []
        else:
            rows = []
        hits: List[KnowledgeHit] = []
        for row in rows or []:
            if isinstance(row, str):
                hits.append(KnowledgeHit(text=row))
                continue
            if not isinstance(row, dict):
                continue
            text = str(row.get("content") or row.get("text") or row.get("chunk") or "").strip()
            if not text:
                continue
            source = str(row.get("source") or row.get("doc_name") or row.get("title") or "")
            try:
                score = float(row.get("score") or row.get("similarity") or 0)
            except (TypeError, ValueError):
                score = 0.0
            hits.append(KnowledgeHit(text=text, source=source, score=score))
        return hits


def as_chunks(hits: List[KnowledgeHit]) -> List[Any]:
    """把远端命中包装成和本地 KnowledgeBase 相同的 Chunk 形状（text/source）。"""
    from .rag import Chunk, _tokenize

    return [Chunk(hit.text, hit.source, _tokenize(hit.text)) for hit in hits]
