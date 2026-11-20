from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

from PIL import Image


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str


@dataclass
class Badge:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


@dataclass
class MessageItem:
    """一条消息：文本 / 图片 / 语音 / 文件 / 系统。"""
    kind: str = "text"          # text | image | voice | file | system
    text: str = ""
    sender: str = ""             # 群聊里是谁发的
    mention: bool = False        # 是否 @我 / @所有人
    media_path: str = ""         # 图片裁图临时文件
    description: str = ""        # 视觉模型 / 占位描述
    bbox: tuple = ()             # 图片在整窗中的坐标 (x1, y1, x2, y2)

    def prompt_text(self) -> str:
        if self.kind == "text":
            base = self.text
        elif self.kind == "image":
            base = f"[图片]{('：' + self.description) if self.description else ''}"
        elif self.kind == "voice":
            base = f"[语音消息]{('：' + self.description) if self.description else '（未转写）'}"
        elif self.kind == "file":
            base = f"[文件]{('：' + self.description) if self.description else ''}"
        else:
            base = self.text or "[系统消息]"
        if self.sender:
            return f"{self.sender}：{base}"
        return base


class Platform(Protocol):
    name: str
    is_windows: bool

    def find_window(self) -> Optional[WindowInfo]: ...
    def capture(self) -> Optional[Image.Image]: ...
    def click(self, x: float, y: float) -> None: ...
    def get_foreground(self) -> int: ...
    def focus(self, window: WindowInfo) -> bool: ...
    def restore_foreground(self, hwnd: int) -> None: ...

    def set_clipboard(self, text: str) -> bool: ...
    def press_ctrl_v(self) -> None: ...
    def press_enter(self) -> None: ...

    # 结构化通道（可选）：实现后优先于 OCR
    def read_header_text(self) -> Optional[str]: ...
    def read_incoming_items(self) -> Optional[List[MessageItem]]: ...
    def last_outgoing_text(self) -> Optional[str]: ...
    def send_image(self, path: str) -> bool: ...

    def close(self) -> None: ...


@dataclass
class MessageLine:
    text: str
    y: float
    x_center: float
    direction: str
    confidence: float = 1.0


@dataclass
class SendOutcome:
    status: str        # verified | uncertain | failed | aborted
    detail: str = ""
