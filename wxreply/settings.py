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
        current = dict(self.data.get("api_keys") or {})
        for model_id, key in keys.items():
            if key is None:
                continue
            current[str(model_id)] = str(key).strip()
        self.data["api_keys"] = {k: v for k, v in current.items() if v}

    def api_keys(self) -> Dict[str, str]:
        return dict(self.data.get("api_keys") or {})

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
