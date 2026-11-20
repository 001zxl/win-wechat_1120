from __future__ import annotations

import base64
import mimetypes
import os
from typing import Optional

import requests

from ..config import AIConfig


class VisionClient:
    """图片理解：把裁出来的图片块交给视觉模型描述。"""

    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.base_url = config.vision_base_url or config.base_url
        self.model = config.vision_model or config.model
        # 视觉凭据独立：没有单独配置时，只在同一个网关下复用聊天 key；
        # 被显式撤销（revoked 含 "vision"）时一律为空
        self.api_key = config.vision_key()

    @property
    def available(self) -> bool:
        # 必须显式配置 vision_model 才算"能用视觉"：
        # 否则会拿纯文本模型（deepseek-chat）硬解图片，模型只能瞎编"这是一张微信聊天截图…"，
        # 还会让每个循环都把窗口当成新图片，无限重写草稿。
        return bool(self.config.vision_model) and bool(self.api_key) \
            and "REPLACE" not in self.api_key

    def describe(self, image_path: str, question: str = "") -> str:
        if not self.available or not image_path or not os.path.exists(image_path):
            return ""
        try:
            with open(image_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
        except Exception:
            return ""
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question or "这是微信聊天里对方发来的一张图片，请用一两句话描述它的内容（产品、报价单、截图文字等）。"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(
                self.base_url.rstrip("/") + "/chat/completions",
                json=payload, headers=headers, timeout=30,
            )
            if response.status_code >= 400:
                return ""
            data = response.json()
            return ((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        except Exception:
            return ""
