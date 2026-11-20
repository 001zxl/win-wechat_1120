"""WeKnora 知识库适配器（可选后端）。

契约按官方机器接口实现（部署版本不同则只需改本文件）：

    POST {base_url}/api/v1/knowledge-search
    X-API-Key: <api_key>
    {"query": "...", "knowledge_base_ids": ["..."], "top_k": 5}

注意：`/api/v1/knowledge/search` 是"文件搜索"，不是知识片段检索，别混用。
远端不可用/无命中时由 agent 回退本地 BM25，并把 last_error 显示到控制台。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import requests


@dataclass
class KnowledgeHit:
    text: str
    source: str = ""          # 文件名 / 标题（展示与审计用）
    score: float = 0.0
    chunk_id: str = ""
    knowledge_id: str = ""
    kb_id: str = ""
    title: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_evidence(self) -> Dict[str, Any]:
        return {"source": self.source or self.title, "score": self.score,
                "id": self.chunk_id or self.knowledge_id, "title": self.title}


class WeKnoraClient:
    def __init__(self, base_url: str, api_key: str = "",
                 search_path: str = "/api/v1/knowledge-search",
                 timeout: float = 8.0, auth_header: str = "X-API-Key",
                 kb_ids: Optional[Iterable[str]] = None, kb_id: str = "",
                 tenant_id: str = "", min_score: float = 0.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.search_path = search_path or "/api/v1/knowledge-search"
        self.timeout = timeout
        self.auth_header = (auth_header or "X-API-Key").strip()
        self.kb_ids = [str(item) for item in (kb_ids or []) if str(item).strip()]
        if not self.kb_ids and kb_id:
            self.kb_ids = [str(kb_id)]
        self.tenant_id = tenant_id or ""
        self.min_score = max(0.0, float(min_score))
        self.last_error = ""
        self.last_status = 0
        self.last_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    # ---------- 请求 ----------
    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            if self.auth_header.lower() == "authorization":
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                headers[self.auth_header] = self.api_key     # 官方：X-API-Key = 原始 key
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id           # 平台级 key 必须带，空间级 key 可省
        return headers

    def payload(self, query: str, top_k: int) -> Dict[str, Any]:
        # 官方字段是 match_count（没有 top_k），上限 200
        body: Dict[str, Any] = {"query": query, "match_count": max(1, min(200, int(top_k)))}
        if self.kb_ids:
            # 明确指定知识范围：不传等于"全库检索"，正式环境必须传
            body["knowledge_base_ids"] = list(self.kb_ids)
        return body

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeHit]:
        self.last_error = ""
        self.last_status = 0
        self.last_count = 0
        if not self.enabled or not (query or "").strip():
            return []
        url = self.base_url + self.search_path
        try:
            response = requests.post(url, json=self.payload(query, top_k),
                                     headers=self.headers(), timeout=self.timeout)
            self.last_status = int(getattr(response, "status_code", 0) or 0)
            if self.last_status >= 400:
                self.last_error = self._describe_error(response)
                return []
            data = response.json()
        except Exception as exc:
            self.last_error = str(exc)
            return []
        if isinstance(data, dict) and data.get("success") is False:
            self.last_error = self._describe_error(data)
            return []
        hits = self.parse_hits(data, min_score=self.min_score)
        self.last_count = len(hits)
        return hits

    @staticmethod
    def _describe_error(payload: Any) -> str:
        """HTTP 状态码是权威的；错误体里带 code/message，方便区分 401 和 403。"""
        if hasattr(payload, "status_code"):
            status = int(getattr(payload, "status_code", 0) or 0)
            try:
                payload = payload.json()
            except Exception:
                return f"HTTP {status}"
        else:
            status = 0
        hint = {401: "API Key 无效", 403: "无检索权限或超出知识库范围",
                404: "检索路径不存在", 409: "缺少 X-Tenant-ID"}.get(status, "")
        message = ""
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "")
            elif error:
                message = str(error)
            elif payload.get("message"):
                message = str(payload["message"])
        parts = [part for part in (f"HTTP {status}" if status else "", hint, message) if part]
        return "：".join(parts) if parts else "未知错误"

    # ---------- 响应解析 ----------
    @classmethod
    def parse_hits(cls, data: Any, min_score: float = 0.0) -> List[KnowledgeHit]:
        rows = cls._find_rows(data)
        hits: List[KnowledgeHit] = []
        for row in rows or []:
            if isinstance(row, str):
                text = row.strip()
                if text:
                    hits.append(KnowledgeHit(text=text))
                continue
            if not isinstance(row, dict):
                continue
            text = cls._first(row, ("content", "text", "chunk", "chunk_content",
                                    "content_text", "passage"))
            text = str(text or "").strip()
            if not text:
                continue
            score = cls._score(row)
            title = str(cls._first(row, ("knowledge_title", "title", "doc_title")) or "")
            filename = str(cls._first(row, ("knowledge_filename", "filename", "file_name",
                                            "doc_name", "source")) or "")
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            hits.append(KnowledgeHit(
                text=text,
                source=filename or str(meta.get("source") or ""),
                score=score,
                chunk_id=str(cls._first(row, ("chunk_id", "id", "chunkId")) or ""),
                knowledge_id=str(cls._first(row, ("knowledge_id", "knowledgeId", "doc_id")) or ""),
                kb_id=str(cls._first(row, ("knowledge_base_id", "kb_id", "knowledgeBaseId")) or ""),
                title=title,
                meta=meta,
            ))
        return cls._filter_scores(hits, min_score)

    @classmethod
    def _find_rows(cls, data: Any, depth: int = 0) -> List[Any]:
        """兼容几种信封：{data:[...]} / {data:{items:[...]}} / {results:[...]} / 直接数组。"""
        if depth > 3 or data is None:
            return []
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        for key in ("results", "hits", "items", "records", "chunks", "list"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        for key in ("data", "result", "payload"):
            if key in data:
                rows = cls._find_rows(data[key], depth + 1)
                if rows:
                    return rows
        # 只有一个列表字段时也认（不同版本字段名可能不同）
        lists = [value for value in data.values() if isinstance(value, list)]
        if len(lists) == 1:
            return lists[0]
        return []

    @staticmethod
    def _first(row: Dict[str, Any], names: Iterable[str]) -> Any:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _score(row: Dict[str, Any]) -> float:
        for name in ("score", "similarity", "relevance_score", "relevance", "rank_score"):
            if name in row:
                try:
                    return float(row[name])
                except (TypeError, ValueError):
                    continue
        return 0.0

    @staticmethod
    def _filter_scores(hits: List[KnowledgeHit], min_score: float) -> List[KnowledgeHit]:
        """不把"返回了任意片段"当成"有依据"：0 分/低于阈值的片段直接丢掉。

        有些部署不做 rerank，所有 score 都是 0 —— 这种情况保留片段，但 score 记 0，
        由业务侧的关键词覆盖判断兜底。
        """
        if not hits:
            return []
        top = max(hit.score for hit in hits)
        threshold = max(0.0, float(min_score))
        kept: List[KnowledgeHit] = []
        for hit in hits:
            if threshold > 0 and hit.score < threshold:
                continue
            if top > 0 and hit.score <= 0:
                continue
            kept.append(hit)
        return kept


def as_chunks(hits: List[KnowledgeHit]) -> List[Any]:
    """把远端命中包装成和本地 KnowledgeBase 相同的 Chunk 形状（含分数/来源）。"""
    from .rag import Chunk, _tokenize

    return [Chunk(hit.text, hit.source or hit.title, _tokenize(hit.text),
                  score=hit.score,
                  chunk_id=hit.chunk_id or hit.knowledge_id,
                  meta={"title": hit.title, "kb_id": hit.kb_id, **hit.meta})
            for hit in hits]
