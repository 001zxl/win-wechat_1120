"""双轨视觉感知：像素哨兵 + 真彩气泡 + OCR 降采样。

左轨：侧边栏红点/像素哈希，决定要不要 OCR。
右轨：聊天区像素哈希 + 真彩气泡判定（绿底=我方），再从最后一条我方消息回溯未读。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .chat import ChatLine, aggregate_incoming, lines_from_ocr
from .layout import Regions
from .ocr import OcrLine

MAX_OCR_WIDTH = 1000


@dataclass
class FrameSignature:
    chat_hash: str = ""
    sidebar_hash: str = ""
    has_red: bool = False
    at: float = 0.0


@dataclass
class PixelSentinel:
    """像素级动态感知：画面未变且无红点时跳过 OCR，降低空闲 CPU。"""

    force_interval_s: float = 1.0
    last: FrameSignature = field(default_factory=FrameSignature)

    def signature(self, image: Image.Image, regions: Regions) -> FrameSignature:
        chat = image.crop(regions.messages).convert("RGB")
        sidebar = image.crop(regions.session_list).convert("RGB")
        red = _red_mask(np.array(sidebar))
        return FrameSignature(
            chat_hash=hashlib.md5(chat.tobytes()).hexdigest(),
            sidebar_hash=hashlib.md5(sidebar.tobytes()).hexdigest(),
            has_red=int(red.sum()) >= 15,
            at=time.time(),
        )

    def should_scan(self, image: Image.Image, regions: Regions, force: bool = False) -> bool:
        if force:
            return True
        sig = self.signature(image, regions)
        previous = self.last
        self.last = sig
        if not previous.at:
            return True
        if sig.chat_hash == previous.chat_hash and sig.sidebar_hash == previous.sidebar_hash:
            return sig.has_red or (sig.at - previous.at) >= self.force_interval_s
        return True


def _red_mask(array: np.ndarray) -> np.ndarray:
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    return (red >= 170) & (green <= 115) & (blue <= 115) & ((red - green) >= 60)


def green_mask(array: np.ndarray) -> np.ndarray:
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    return (green > red + 18) & (green > blue + 18) & (green > 45)


def _background_is_green(image: Image.Image, box: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = [int(v) for v in box]
    pad_x = max(3, int((x2 - x1) * 0.08))
    pad_y = max(2, int((y2 - y1) * 0.35))
    left = max(0, x1 - pad_x)
    right = min(image.width, x2 + pad_x)
    top = max(0, y1 - pad_y)
    bottom = min(image.height, y2 + pad_y)
    if right - left < 4 or bottom - top < 4:
        return False
    patch = np.array(image.crop((left, top, right, bottom)).convert("RGB"))
    mask = green_mask(patch)
    return float(mask.mean()) >= 0.28


def downscale_for_ocr(image: Image.Image, max_width: int = MAX_OCR_WIDTH) -> Tuple[Image.Image, float]:
    if image.width <= max_width + 100:
        return image, 1.0
    ratio = max_width / float(image.width)
    resized = image.resize((max_width, max(1, int(image.height * ratio))), Image.BILINEAR)
    return resized, ratio


def lines_from_ocr_colored(
    image: Image.Image,
    regions: Regions,
    ocr_lines: List[OcrLine],
    min_confidence: float = 0.5,
    scale: float = 1.0,
) -> List[ChatLine]:
    """真彩气泡优先：绿底判我方，其余按左右位置；坐标按 scale 还原到整窗。"""
    chat_left, _top, chat_right, _bottom = regions.chat
    width = max(1, chat_right - chat_left)
    result: List[ChatLine] = []
    for line in ocr_lines:
        if line.score < min_confidence or not (line.text or "").strip():
            continue
        x1, y1, x2, y2 = [v / scale for v in line.bbox]
        x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if _background_is_green(image, (x1, y1, x2, y2)):
            direction = "out"
        else:
            ratio = (x_center - chat_left) / width
            direction = "out" if ratio >= 0.55 else "in"
        result.append(ChatLine(line.text.strip(), y_center, x_center, direction))
    result.sort(key=lambda item: item.y)
    return result


def pending_incoming(image: Image.Image, regions: Regions, lines: List[ChatLine]) -> List[str]:
    """从最后一条我方气泡回溯，聚合对方未读。"""
    if not lines:
        return []
    last_out = -1
    for index, line in enumerate(lines):
        if line.direction == "out":
            last_out = index
    incoming = [line.text for line in lines[last_out + 1:] if line.direction == "in"]
    merged: List[str] = []
    for text in incoming:
        if merged and len(text) <= 10:
            merged[-1] = merged[-1] + "，" + text
        else:
            merged.append(text)
    return merged


def extract_pending(image: Image.Image, regions: Regions, ocr_lines: List[OcrLine], scale: float = 1.0) -> List[str]:
    lines = lines_from_ocr_colored(image, regions, ocr_lines, scale=scale)
    return pending_incoming(image, regions, lines)
