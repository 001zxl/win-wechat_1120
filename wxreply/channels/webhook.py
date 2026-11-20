from __future__ import annotations

import json
import urllib.request
from typing import Dict, Optional

from .base import Channel, ChannelCapabilities


class WebhookChannel(Channel):
    """LINE / Telegram / 通用 Webhook 通道。

    options:
      endpoint   完整 URL（LINE 填 https://api.line.me/v2/bot/message/push）
      token      Bearer token（Telegram 可留空，endpoint 里已含 bot token）
      provider   line | telegram | generic
      target_key 接收者字段名：line 用 to，telegram 用 chat_id，generic 用 to
    """

    type = "webhook"
    capabilities = ChannelCapabilities(
        can_send=True, can_read_history=False, has_avatars=False,
        supports_groups=True, transport="bot-api",
    )

    def __init__(self, channel_id: str, options: Optional[Dict[str, str]] = None) -> None:
        self.id = channel_id
        self.options = dict(options or {})
        self.provider = self.options.get("provider", "generic").lower()

    def _payload(self, peer: str, text: str) -> dict:
        if self.provider == "line":
            key = self.options.get("target_key", "to")
            recipient = peer
            if peer.startswith("group:"):
                key, recipient = "groupId", peer.split(":", 1)[1]
            elif peer.startswith("room:"):
                key, recipient = "roomId", peer.split(":", 1)[1]
            return {key: recipient, "messages": [{"type": "text", "text": text}]}
        if self.provider == "telegram":
            return {"chat_id": peer, "text": text}
        key = self.options.get("target_key", "to")
        return {key: peer, "text": text}

    def send(self, peer: str, text: str) -> bool:
        endpoint = self.options.get("endpoint", "")
        if not endpoint or not text.strip():
            return False
        body = json.dumps(self._payload(peer, text), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = self.options.get("token", "")
        if token and self.provider == "line":
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def health(self) -> Dict[str, object]:
        return {"id": self.id, "type": self.type, "ok": bool(self.options.get("endpoint")),
                "provider": self.provider}
