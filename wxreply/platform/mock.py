from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .base import MessageItem, WindowInfo


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def norm_message(raw: Any) -> Dict[str, Any]:
    """兼容旧的 (dir, text) 元组和新的 dict 消息。"""
    if isinstance(raw, dict):
        return {
            "dir": raw.get("dir", "in"),
            "kind": raw.get("kind", "text"),
            "text": raw.get("text", ""),
            "sender": raw.get("sender", ""),
            "mention": bool(raw.get("mention", False)),
            "description": raw.get("description", ""),
        }
    direction, text = raw
    return {"dir": direction, "kind": "text", "text": text, "sender": "",
            "mention": False, "description": ""}


@dataclass
class MockSession:
    name: str
    unread: int = 0
    messages: List[Any] = field(default_factory=list)


class MockPlatform:
    """在 macOS/CI 上验证全链路的假微信：渲染合成窗口 + 结构化读数。"""

    name = "mock"
    is_windows = False

    WIDTH = 1160
    HEIGHT = 760

    def __init__(self) -> None:
        self.window = WindowInfo(hwnd=1001, title="微信", class_name="Qt51514QWindowIcon")
        self.sessions: List[MockSession] = [
            MockSession("张三", unread=2, messages=[("in", "在吗？"), ("in", "这款怎么卖？")]),
            MockSession("李四", unread=0, messages=[("in", "上次的样品收到了"), ("out", "好的，有问题随时说～")]),
            MockSession("外贸客户-A", unread=1, messages=[("in", "请报个价，5000 件")]),
            MockSession("产品交流群(25)", unread=3, messages=[
                {"dir": "in", "kind": "text", "text": "群里有人吗", "sender": "阿强"},
                {"dir": "in", "kind": "text", "text": "@我 这个多少钱", "sender": "小美", "mention": True},
            ]),
        ]
        self.current: Optional[int] = None
        self.input_focused = False
        self.clipboard = ""
        self.draft = ""
        self.foreground = 0
        self.sent: List[Tuple[str, str]] = []
        self._font_small = _load_font(14)
        self._font_name = _load_font(15)
        self._font_chat = _load_font(14)

    # ---------- 几何 ----------
    def _metrics(self) -> Dict[str, float]:
        w, h = self.WIDTH, self.HEIGHT
        nav = 64.0
        list_left, list_right = nav, w * 0.34
        return {
            "nav": nav,
            "list_left": list_left,
            "list_right": list_right,
            "list_top": 80.0,
            "row_h": 70.0,
            "chat_left": list_right,
            "chat_right": w,
            "header_bottom": h * 0.09,
            "input_top": h * 0.78,
            "height": h,
            "width": w,
        }

    # ---------- 渲染 ----------
    def capture(self) -> Optional[Image.Image]:
        m = self._metrics()
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT), (237, 237, 237))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, m["nav"], m["height"]], fill=(46, 46, 46))
        draw.ellipse([18, 24, 46, 52], fill=(120, 200, 120))
        draw.rectangle([m["list_left"], 0, m["list_right"], m["height"]], fill=(245, 245, 245))
        draw.rectangle([m["list_left"], 58, m["list_right"] - 8, 76], fill=(228, 228, 228))
        for index, session in enumerate(self.sessions):
            top = m["list_top"] + index * m["row_h"]
            if index == self.current:
                draw.rectangle([m["list_left"], top, m["list_right"], top + m["row_h"]], fill=(214, 214, 214))
            draw.ellipse([m["list_left"] + 14, top + 16, m["list_left"] + 54, top + 56], fill=(190, 190, 190))
            draw.text((m["list_left"] + 66, top + 22), session.name, font=self._font_name, fill=(20, 20, 20))
            if session.unread:
                cx, cy = m["list_left"] + 46, top + 58
                draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(230, 40, 40))
                draw.text((cx - 5, cy - 8), str(session.unread), font=self._font_small, fill=(255, 255, 255))
        draw.rectangle([m["chat_left"], 0, m["chat_right"], m["height"]], fill=(242, 242, 242))
        if self.current is not None:
            session = self.sessions[self.current]
            draw.rectangle([m["chat_left"], 0, m["chat_right"], m["header_bottom"]], fill=(247, 247, 247))
            draw.text((m["chat_left"] + 24, m["header_bottom"] - 30), session.name,
                      font=self._font_name, fill=(20, 20, 20))
            y = m["header_bottom"] + 24
            for raw in session.messages[-8:]:
                message = norm_message(raw)
                kind = message["kind"]
                text = message["text"]
                if kind == "image":
                    text = "[图片]"
                elif kind == "voice":
                    text = "[语音]"
                elif kind == "file":
                    text = "[文件]"
                bubble_w = 120 if kind in ("image", "voice", "file") else min(420, 40 + len(text) * 15)
                if message["sender"]:
                    draw.text((m["chat_left"] + 26, y - 2), message["sender"],
                              font=self._font_small, fill=(120, 120, 120))
                    y += 14
                if message["dir"] == "in":
                    draw.rounded_rectangle([m["chat_left"] + 24, y, m["chat_left"] + 24 + bubble_w, y + 44],
                                           radius=8, fill=(255, 255, 255))
                    draw.text((m["chat_left"] + 36, y + 12), text, font=self._font_chat, fill=(30, 30, 30))
                else:
                    right = m["chat_right"] - 24
                    draw.rounded_rectangle([right - bubble_w, y, right, y + 44], radius=8, fill=(149, 236, 105))
                    draw.text((right - bubble_w + 12, y + 12), text, font=self._font_chat, fill=(20, 20, 20))
                y += 60
        else:
            draw.text((m["chat_left"] + 40, m["header_bottom"] + 40), "点击左侧会话开始", font=self._font_chat,
                      fill=(150, 150, 150))
        draw.rectangle([m["chat_left"], m["input_top"], m["chat_right"], m["height"]], fill=(250, 250, 250))
        draw.rectangle([m["chat_left"] + 16, m["input_top"] + 12, m["chat_right"] - 140, m["height"] - 20],
                       fill=(255, 255, 255), outline=(220, 220, 220))
        if self.draft:
            draw.text((m["chat_left"] + 28, m["input_top"] + 30), self.draft, font=self._font_chat, fill=(20, 20, 20))
        draw.text((m["chat_right"] - 120, m["input_top"] + 30), "发送(S)", font=self._font_small, fill=(60, 130, 60))
        return img

    def find_window(self) -> WindowInfo:
        return self.window

    # ---------- 交互 ----------
    def click(self, x: float, y: float) -> None:
        m = self._metrics()
        if m["list_left"] <= x <= m["list_right"] and y >= m["list_top"]:
            index = int((y - m["list_top"]) // m["row_h"])
            if 0 <= index < len(self.sessions):
                self.current = index
                self.sessions[index].unread = 0
                self.input_focused = False
            return
        if x >= m["chat_left"] and y >= m["input_top"]:
            self.input_focused = True

    def get_foreground(self) -> int:
        return self.foreground

    def focus(self, window: WindowInfo) -> bool:
        self.foreground = window.hwnd
        return True

    def restore_foreground(self, hwnd: int) -> None:
        self.foreground = hwnd

    def get_clipboard(self) -> str:
        return self.clipboard

    def set_clipboard(self, text: str) -> bool:
        self.clipboard = text
        return True

    def press_ctrl_v(self) -> None:
        if self.input_focused:
            self.draft += self.clipboard

    def press_enter(self) -> None:
        if self.current is None:
            return
        if self.draft:
            session = self.sessions[self.current]
            session.messages.append({"dir": "out", "kind": "text", "text": self.draft})
            self.sent.append((session.name, self.draft))
            self.draft = ""
            self.input_focused = False

    # ---------- 结构化通道 ----------
    def read_header_text(self) -> Optional[str]:
        if self.current is None:
            return None
        return self.sessions[self.current].name

    def read_incoming_items(self) -> Optional[List[MessageItem]]:
        if self.current is None:
            return None
        messages = [norm_message(raw) for raw in self.sessions[self.current].messages]
        last_out = -1
        for index, message in enumerate(messages):
            if message["dir"] == "out":
                last_out = index
        items: List[MessageItem] = []
        for message in messages[last_out + 1:]:
            if message["dir"] == "out":
                continue
            items.append(MessageItem(
                kind=message["kind"], text=message["text"], sender=message["sender"],
                mention=message["mention"], description=message["description"],
            ))
        return items

    def last_outgoing_text(self) -> Optional[str]:
        if self.current is None:
            return None
        for raw in reversed(self.sessions[self.current].messages):
            message = norm_message(raw)
            if message["dir"] == "out":
                return message["text"]
        return None

    def send_image(self, path: str) -> bool:
        if self.current is None:
            return False
        self.sessions[self.current].messages.append(
            {"dir": "out", "kind": "image", "text": "", "description": path}
        )
        return True

    def close(self) -> None:
        return None

    def unread_targets(self):
        """虚拟平台：直接返回未读会话的点击坐标，避免对合成窗口 OCR。"""
        m = self._metrics()
        result = []
        for index, session in enumerate(self.sessions):
            if not session.unread:
                continue
            x = m["list_left"] + (m["list_right"] - m["list_left"]) * 0.55
            y = m["list_top"] + index * m["row_h"] + 58
            result.append((session.name, float(x), float(y)))
        return result

    def session_snapshot(self):
        """虚拟微信的会话列表（供控制台同步，不走真实微信）。"""
        result = []
        for session in self.sessions:
            preview = ""
            if session.messages:
                last = norm_message(session.messages[-1])
                if last["kind"] == "text":
                    preview = last["text"]
                elif last["kind"] == "image":
                    preview = "[图片]"
                elif last["kind"] == "voice":
                    preview = "[语音]"
                else:
                    preview = "[文件]"
                if last["dir"] == "out":
                    preview = "我方：" + preview
            result.append({"name": session.name, "preview": preview[:30],
                           "unread": session.unread, "time": "刚刚"})
        return result

    # ---------- 测试辅助 ----------
    def force_open(self, name: str) -> None:
        for index, session in enumerate(self.sessions):
            if session.name == name:
                self.current = index
                session.unread = 0
                return
        self.sessions.append(MockSession(name, unread=1, messages=[("in", "在吗")]))
        self.current = len(self.sessions) - 1

    def unread_badges(self) -> List[Tuple[str, Tuple[float, float]]]:
        m = self._metrics()
        result = []
        for index, session in enumerate(self.sessions):
            if session.unread:
                y = m["list_top"] + index * m["row_h"] + 58
                result.append((session.name, (m["list_left"] + 46, y)))
        return result
