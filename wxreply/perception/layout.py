from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from PIL import Image

Box = Tuple[int, int, int, int]


@dataclass
class Regions:
    width: int
    height: int
    nav: Box
    session_list: Box
    chat: Box
    header: Box
    messages: Box
    input: Box
    send_button: Box

    def crop(self, image: Image.Image, name: str) -> Image.Image:
        box = getattr(self, name)
        return image.crop(box)

    def to_dict(self) -> Dict[str, Box]:
        return {
            "nav": self.nav,
            "session_list": self.session_list,
            "chat": self.chat,
            "header": self.header,
            "messages": self.messages,
            "input": self.input,
            "send_button": self.send_button,
        }


class LayoutCalculator:
    """按窗口比例计算区域，换版本只调比例。"""

    def __init__(
        self,
        nav_ratio: float = 0.055,
        list_ratio: float = 0.34,
        header_ratio: float = 0.09,
        input_ratio: float = 0.22,
        overrides: Dict[str, float] = None,
    ) -> None:
        self.nav_ratio = nav_ratio
        self.list_ratio = list_ratio
        self.header_ratio = header_ratio
        self.input_ratio = input_ratio
        self.overrides = overrides or {}

    def _ratio(self, key: str, default: float) -> float:
        return float(self.overrides.get(key, default))

    def regions(self, width: int, height: int) -> Regions:
        width, height = int(width), int(height)
        nav_right = int(width * self._ratio("nav_ratio", self.nav_ratio))
        list_right = int(width * self._ratio("list_ratio", self.list_ratio))
        chat_left = int(width * self._ratio("chat_left", list_right / max(1, width)))
        header_bottom = int(height * self._ratio("header_ratio", self.header_ratio))
        input_top = int(height * self._ratio("input_top", 1.0 - self.input_ratio))
        # 微信 4.x 的输入框（含工具栏）高度基本固定约 130px；窗口变矮时按比例算会落到工具栏/空白上，
        # 这里夹一下，保证 input 区域始终压住真正的输入框
        input_top = min(input_top, max(int(height * 0.45), height - 130))
        send_left = int(width * self._ratio("send_left", 0.88))
        return Regions(
            width=width,
            height=height,
            nav=(0, 0, nav_right, height),
            session_list=(nav_right, 0, list_right, height),
            chat=(chat_left, 0, width, height),
            header=(chat_left, 0, width, header_bottom),
            messages=(chat_left, header_bottom, width, input_top),
            input=(chat_left, input_top, width, height),
            send_button=(send_left, input_top, width, height),
        )
