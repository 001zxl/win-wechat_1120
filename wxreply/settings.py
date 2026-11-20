from __future__ import annotations

import json
import os
from typing import Any, Dict, List


class SettingsStore:
    """控制台写回的本地设置：模型 Key / 白名单 / 布局。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)

    # ---------- API Key ----------
    def set_api_keys(self, keys: Dict[str, str]) -> None:
        """写入/删除模型 Key。空字符串 = 明确删除（不是"不修改"）。"""
        current = dict(self.data.get("api_keys") or {})
        revoked = [str(item) for item in (self.data.get("revoked") or []) if str(item).strip()]
        for model_id, key in keys.items():
            if key is None:
                continue
            model_id = str(model_id)
            if str(key).strip():
                current[model_id] = str(key).strip()
                revoked = [item for item in revoked if item.lower() != model_id.lower()]
            else:
                current.pop(model_id, None)
                if model_id.lower() not in {item.lower() for item in revoked}:
                    revoked.append(model_id)
        self.data["api_keys"] = current
        self.data["revoked"] = revoked

    def revoke(self, names: List[str]) -> List[str]:
        """显式撤销：删掉持久化的 key 并记下"已撤销"，防止从顶层配置继承回来。

        "*" 会展开成"当前所有已保存的模型 key + 顶层/视觉/Embedding"，
        这样用户之后重新录入某个模型的 key 仍然有效，不会被一个通配符永久锁死。
        """
        current = dict(self.data.get("api_keys") or {})
        revoked = {str(item).strip().lower(): str(item).strip()
                   for item in (self.data.get("revoked") or []) if str(item).strip()}
        for name in names or []:
            token = str(name or "").strip()
            if not token:
                continue
            if token == "*":
                expanded = list(current.keys()) + ["default", "vision", "embedding"]
                for item in expanded:
                    revoked[item.lower()] = item
                current = {}
                continue
            revoked[token.lower()] = token
            current = {key: value for key, value in current.items()
                       if key.strip().lower() != token.lower()}
        self.data["api_keys"] = current
        self.data["revoked"] = sorted(revoked.values())
        return list(self.data["revoked"])

    def restore(self, names: List[str]) -> List[str]:
        """解除撤销标记（重新录入 key 时调用）。"""
        drop = {str(name or "").strip().lower() for name in names or [] if str(name).strip()}
        if "*" in drop:
            remaining: List[str] = []
        else:
            remaining = [str(item) for item in (self.data.get("revoked") or [])
                         if str(item).strip().lower() not in drop]
        self.data["revoked"] = remaining
        return remaining

    def api_keys(self) -> Dict[str, str]:
        return dict(self.data.get("api_keys") or {})

    def revoked(self) -> List[str]:
        return [str(item) for item in (self.data.get("revoked") or []) if str(item).strip()]

    # ---------- 白名单 ----------
    def set_whitelist(self, names: List[str]) -> None:
        self.data["whitelist"] = [str(name).strip() for name in names if str(name).strip()]

    def whitelist(self) -> List[str]:
        return list(self.data.get("whitelist") or [])

    # ---------- 布局 ----------
    def set_layout(self, ratios: Dict[str, float]) -> None:
        current = dict(self.data.get("layout") or {})
        for key in ("nav_ratio", "list_ratio", "header_ratio", "input_ratio"):
            if key in ratios:
                try:
                    current[key] = round(float(ratios[key]), 4)
                except (TypeError, ValueError):
                    continue
        self.data["layout"] = current

    def layout(self) -> Dict[str, float]:
        return dict(self.data.get("layout") or {})

    # ---------- 视觉/向量模型 ----------
    def set_models(self, vision_model: str = None, embedding_model: str = None) -> None:
        if vision_model is not None:
            self.data["vision_model"] = str(vision_model).strip()
        if embedding_model is not None:
            self.data["embedding_model"] = str(embedding_model).strip()
