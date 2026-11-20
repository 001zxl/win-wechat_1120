"""入站 Webhook 归一化：把 LINE / Telegram / 通用 JSON 统一成 IncomingMessage。"""
from __future__ import annotations

from typing import Optional

from .base import IncomingMessage


def parse_inbound(channel_type: str, payload: dict) -> Optional[IncomingMessage]:
    kind = (channel_type or "").lower()
    if not isinstance(payload, dict):
        return None
    if kind == "line":
        for event in payload.get("events") or []:
            if not isinstance(event, dict) or event.get("type") != "message":
                continue
            message = event.get("message") or {}
            if message.get("type") != "text":
                continue
            source = event.get("source") or {}
            peer = source.get("userId")
            if source.get("groupId"):
                peer = "group:" + str(source["groupId"])
            elif source.get("roomId"):
                peer = "room:" + str(source["roomId"])
            text = (message.get("text") or "").strip()
            if peer and text:
                return IncomingMessage("line", str(peer), text, raw=event)
        return None
    if kind == "telegram":
        message = (payload.get("message") or payload.get("edited_message") or {})
        chat = message.get("chat") or {}
        text = (message.get("text") or "").strip()
        peer = chat.get("id")
        if peer is not None and text:
            return IncomingMessage("telegram", str(peer), text, raw=message)
        return None
    text = str(payload.get("text") or payload.get("message") or "").strip()
    peer = payload.get("peer") or payload.get("from") or payload.get("user") or payload.get("chat_id")
    if peer is not None and text:
        return IncomingMessage(kind or "generic", str(peer), text, raw=payload)
    return None
