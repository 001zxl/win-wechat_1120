from __future__ import annotations

from typing import Dict

from .base import Channel, ChannelCapabilities


class WeChatChannel(Channel):
    """把现有微信视觉/RPA 后端包装成通道（电脑操作型）。"""

    type = "wechat"
    capabilities = ChannelCapabilities(
        can_send=True, can_read_history=True, has_avatars=True,
        supports_groups=True, transport="computer-use",
    )

    def __init__(self, agent, channel_id: str = "wechat") -> None:
        self.agent = agent
        self.id = channel_id

    def send(self, peer: str, text: str) -> bool:
        result = self.agent.send_manual(peer, text)
        return bool(result and result.ok)

    def health(self) -> Dict[str, object]:
        platform = getattr(self.agent.platform, "name", "unknown")
        return {"id": self.id, "type": self.type, "ok": True, "platform": platform}
