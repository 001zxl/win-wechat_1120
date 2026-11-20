"""统一 IM 通道抽象：一套助理接入多个 IM 平台。

一个 Channel = 收消息 + 发消息 + 能力声明。WeChat 用视觉/RPA 适配器，
LINE/Telegram/DingTalk 等走 Webhook/HTTP Bot API，共用同一套记忆与人设。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IncomingMessage:
    channel: str
    peer: str
    text: str
    raw: Optional[dict] = None


@dataclass
class ChannelCapabilities:
    can_send: bool = True
    can_read_history: bool = False
    has_avatars: bool = False
    supports_groups: bool = False
    transport: str = "webhook"   # webhook | computer-use | bot-api


@dataclass
class ChannelConfig:
    id: str
    type: str
    enabled: bool = True
    options: Dict[str, str] = field(default_factory=dict)


class Channel:
    id: str = ""
    type: str = ""
    capabilities = ChannelCapabilities()

    def send(self, peer: str, text: str) -> bool:
        raise NotImplementedError

    def health(self) -> Dict[str, object]:
        return {"id": self.id, "type": self.type, "ok": True}

    def close(self) -> None:
        pass
