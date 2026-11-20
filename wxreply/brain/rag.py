from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import requests

_TOKEN_ALNUM = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class Chunk:
    text: str
    source: str
    tokens: List[str]
    # 检索结果的溯源信息：分数、片段 ID、知识库元数据（标题/文件名/版本等）
    score: float = 0.0
    chunk_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def as_evidence(self) -> Dict[str, Any]:
        return {"source": self.source, "score": self.score, "id": self.chunk_id,
                "title": str((self.meta or {}).get("title", ""))}


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = list(_TOKEN_ALNUM.findall(text))
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(chinese)
    tokens.extend("".join(pair) for pair in zip(chinese, chinese[1:]))
    return tokens


_HEADING = re.compile(r"\n(?=#{1,6}\s)")


def _split(text: str, max_chars: int = 400) -> List[str]:
    """按 Markdown 标题切块，再按空行/句号合并。

    不能把整份 FAQ 合成一个大块：那样"任意片段"都会命中所有主题，
    依据判断就形同虚设（复查第 7 条）。
    """
    chunks: List[str] = []
    for block in _HEADING.split(text or ""):
        block = block.strip()
        if not block:
            continue
        parts = [part.strip() for part in re.split(r"\n{2,}|(?<=。)", block) if part.strip()]
        current = ""
        for part in parts:
            if len(current) + len(part) <= max_chars:
                current += part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
    return chunks


def _ratio(scores: List[float], index: int) -> float:
    """BM25 分数没有上界：换算成"相对最高分"的 0~1，便于展示和落库。"""
    top = max(scores) if scores else 0.0
    if top <= 0:
        return 0.0
    return max(0.0, scores[index] / top)


class EmbeddingClient:
    """OpenAI 兼容 /embeddings；不可用时自动退回 BM25。"""

    def __init__(self, model: str, base_url: str, api_key: str, cache_path: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        self.failed = False
        self._load_cache()

    @property
    def available(self) -> bool:
        return bool(self.model and self.api_key and "REPLACE" not in self.api_key and not self.failed)

    def _load_cache(self) -> None:
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as handle:
                    self.cache = json.load(handle)
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as handle:
                json.dump(self.cache, handle)
        except Exception:
            pass

    @property
    def namespace(self) -> str:
        # 缓存必须绑定"供应商 + 模型 + 维度"：换 embedding 模型后旧向量不可复用
        return f"{self.base_url}|{self.model}"

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.namespace}|{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.available or not texts:
            return None
        missing = [text for text in texts if self._key(text) not in self.cache]
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            try:
                response = requests.post(
                    self.base_url + "/embeddings",
                    json={"model": self.model, "input": batch},
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=20,
                )
                if response.status_code >= 400:
                    self.failed = True
                    return None
                data = response.json().get("data") or []
                for text, row in zip(batch, data):
                    self.cache[self._key(text)] = row.get("embedding") or []
                self._save_cache()
            except Exception:
                self.failed = True
                return None
        vectors = [self.cache.get(self._key(text)) or [] for text in texts]
        if any(not vector for vector in vectors):
            return None
        return vectors


class KnowledgeBase:
    def __init__(self, directory: str, mode: str = "bm25", embeddings: Optional[EmbeddingClient] = None) -> None:
        self.directory = directory
        self.mode = mode
        self.embeddings = embeddings
        self.chunks: List[Chunk] = []
        self._vectors: Optional[List[List[float]]] = None
        self._doc_freq: Dict[str, int] = {}
        self._avg_len = 0.0
        self.min_cosine = 0.30          # 向量相似度低于该值不算"命中"
        self._load()

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def version(self) -> str:
        """知识库版本：内容变化即变化，控制台可以看到当前生效的是哪一版。"""
        digest = hashlib.sha1()
        for chunk in self.chunks:
            digest.update((chunk.source or "").encode("utf-8"))
            digest.update(chunk.text.encode("utf-8"))
        return digest.hexdigest()[:12]

    def reload(self) -> int:
        """重新扫描 kb 目录（文件改了以后手动/定时调用）。"""
        self.chunks = []
        self._doc_freq = {}
        self._avg_len = 0.0
        self._vectors = None
        self._load()
        return len(self.chunks)


    def _load(self) -> None:
        if not self.directory or not os.path.isdir(self.directory):
            return
        for root, _dirs, files in os.walk(self.directory):
            for name in sorted(files):
                if not name.lower().endswith((".md", ".txt")):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        content = handle.read()
                except Exception:
                    continue
                for index, piece in enumerate(_split(content)):
                    source = os.path.relpath(path, self.directory)
                    self.chunks.append(Chunk(
                        piece, source, _tokenize(piece),
                        chunk_id=f"{source}#{index}",
                        meta={"title": os.path.splitext(name)[0]},
                    ))
        if not self.chunks:
            return
        self._avg_len = sum(len(chunk.tokens) for chunk in self.chunks) / len(self.chunks)
        for chunk in self.chunks:
            for token in set(chunk.tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

    def _bm25(self, tokens: List[str]) -> List[float]:
        total = len(self.chunks)
        k1, b = 1.5, 0.75
        scores: List[float] = []
        for chunk in self.chunks:
            counts: Dict[str, int] = {}
            for token in chunk.tokens:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            length = max(1, len(chunk.tokens))
            for token in tokens:
                freq = counts.get(token, 0)
                if not freq:
                    continue
                df = self._doc_freq.get(token, 1)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += idf * (freq * (k1 + 1)) / (
                    freq + k1 * (1 - b + b * length / max(self._avg_len, 1e-9))
                )
            scores.append(score)
        return scores

    def _ensure_vectors(self) -> Optional[List[List[float]]]:
        if self._vectors is not None:
            return self._vectors
        if self.mode != "hybrid" or self.embeddings is None or not self.embeddings.available:
            return None
        vectors = self.embeddings.embed([chunk.text for chunk in self.chunks])
        self._vectors = vectors
        return vectors

    def search(self, query: str, top_k: int = 3) -> List[Chunk]:
        if not self.chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        bm25 = self._bm25(tokens)
        vectors = self._ensure_vectors()
        if vectors is None:
            order = sorted(range(len(self.chunks)), key=lambda i: bm25[i], reverse=True)
            picked = [i for i in order[:top_k] if bm25[i] > 0]
            return [self._with_score(i, _ratio(bm25, i)) for i in picked]
        query_vec = self.embeddings.embed([query]) if self.embeddings else None
        if not query_vec:
            order = sorted(range(len(self.chunks)), key=lambda i: bm25[i], reverse=True)
            picked = [i for i in order[:top_k] if bm25[i] > 0]
            return [self._with_score(i, _ratio(bm25, i)) for i in picked]
        q = query_vec[0]
        cosines = []
        for vector in vectors:
            dot = sum(a * b for a, b in zip(q, vector))
            norm = math.sqrt(sum(a * a for a in q)) * math.sqrt(sum(b * b for b in vector)) or 1e-9
            cosines.append(dot / norm)
        # RRF 融合
        bm25_rank = {i: rank for rank, i in enumerate(sorted(range(len(bm25)), key=lambda i: bm25[i], reverse=True))}
        vec_rank = {i: rank for rank, i in enumerate(sorted(range(len(cosines)), key=lambda i: cosines[i], reverse=True))}
        fused = {i: 1.0 / (60 + bm25_rank[i]) + 1.0 / (60 + vec_rank[i]) for i in range(len(bm25))}
        # 关键词和向量都没命中时返回空：宁可"没有可靠答案"，也不要硬塞一段知识
        eligible = [i for i in range(len(bm25)) if bm25[i] > 0 or cosines[i] >= self.min_cosine]
        if not eligible:
            return []
        order = sorted(eligible, key=lambda i: fused[i], reverse=True)
        return [self._with_score(i, max(0.0, cosines[i])) for i in order[:top_k]]

    def _with_score(self, index: int, score: float) -> Chunk:
        """返回带分数的副本：Chunk 是共享对象，不能就地改分。"""
        return replace(self.chunks[index], score=round(float(score), 4))
