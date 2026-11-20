from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .base import Channel, ChannelConfig
from .wechat import WeChatChannel
from .webhook import WebhookChannel


def contact_key(channel_id: str, peer: str) -> str:
    """跨 IM 共用记忆：同一联系人在不同通道可用 channel:peer 关联。"""
    return f"{channel_id}:{peer}"


class ChannelRegistry:
    def __init__(self) -> None:
        self._channels: Dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        self._channels[channel.id] = channel

    def get(self, channel_id: str) -> Optional[Channel]:
        return self._channels.get(channel_id)

    def all(self) -> List[Channel]:
        return list(self._channels.values())

    def send(self, channel_id: str, peer: str, text: str) -> bool:
        channel = self.get(channel_id)
        if channel is None:
            return False
        return channel.send(peer, text)

    def health(self) -> List[Dict[str, object]]:
        return [channel.health() for channel in self._channels.values()]


def build_registry(configs: Iterable[ChannelConfig], agent=None) -> ChannelRegistry:
    registry = ChannelRegistry()
    for item in configs:
        if not item.enabled:
            continue
        if item.type == "wechat" and agent is not None:
            registry.register(WeChatChannel(agent, item.id or "wechat"))
        elif item.type in ("line", "telegram", "webhook", "dingtalk"):
            registry.register(WebhookChannel(item.id or item.type, item.options))
    return registry
