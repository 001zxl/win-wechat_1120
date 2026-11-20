"""入站路由：通道 → 统一助理 → 原通道回发。"""
from __future__ import annotations

from typing import Any, Dict

from .inbound import parse_inbound
from .registry import contact_key


def route_inbound(registry, agent, channel_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    channel = registry.get(channel_id) if registry is not None else None
    if channel is None:
        return {"ok": False, "error": "channel not configured"}
    message = parse_inbound(channel.type, payload)
    if message is None:
        return {"ok": True, "skipped": True}
    memory_peer = contact_key(channel_id, message.peer)
    result = agent.handle_inbound(memory_peer, message.text, auto=True)
    if result is None:
        return {"ok": False, "error": "reply_failed"}
    sent = channel.send(message.peer, result["reply"])
    return {"ok": True, "sent": bool(sent), "reply": result["reply"]}
