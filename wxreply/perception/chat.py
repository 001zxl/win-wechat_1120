from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from .layout import Regions
from .ocr import OcrLine


@dataclass
class ChatLine:
    text: str
    y: float
    x_center: float
    direction: str  # in | out


def lines_from_ocr(
    ocr_lines: List[OcrLine], regions: Regions, min_confidence: float = 0.5
) -> List[ChatLine]:
    chat_left, _top, chat_right, _bottom = regions.chat
    width = max(1, chat_right - chat_left)
    result: List[ChatLine] = []
    for line in ocr_lines:
        if line.score < min_confidence or not line.text.strip():
            continue
        x_center, y_center = line.center
        ratio = (x_center - chat_left) / width
        direction = "out" if ratio >= 0.55 else "in"
        result.append(ChatLine(line.text.strip(), y_center, x_center, direction))
    result.sort(key=lambda item: item.y)
    return result


def aggregate_incoming(lines: List[ChatLine]) -> List[str]:
    """只取最后一条我方消息之后的对方消息（连发聚合）。"""
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


def region_hash(image: Image.Image, regions: Regions) -> str:
    crop = image.crop(regions.messages).convert("RGB")
    return hashlib.md5(crop.tobytes()).hexdigest()


def settle_capture(
    capture: Callable[[], Optional[Image.Image]],
    regions: Regions,
    settle_s: float = 0.35,
    max_wait_s: float = 1.2,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[Image.Image]:
    """像素沉降：画面稳定即返回，最多等 max_wait_s（对方连发时继续等）。"""
    image = capture()
    if image is None:
        return None
    current = region_hash(image, regions)
    waited = 0.0
    while waited < max_wait_s:
        sleep(settle_s)
        waited += settle_s
        latest = capture()
        if latest is None:
            break
        latest_hash = region_hash(latest, regions)
        if latest_hash == current:
            return latest
        image, current = latest, latest_hash
    return image


def changed_ratio(before: Image.Image, after: Image.Image, regions: Regions) -> float:
    a = np.array(before.crop(regions.messages).convert("RGB"), dtype=np.int16)
    b = np.array(after.crop(regions.messages).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return 1.0
    diff = np.abs(a - b).sum(axis=2) > 30
    return float(diff.mean()) if diff.size else 0.0


def has_new_outgoing(image: Image.Image, regions: Regions) -> bool:
    """检测消息区右下是否出现绿色气泡（我方发送成功）。"""
    crop = np.array(image.crop(regions.messages).convert("RGB"))
    height, width = crop.shape[:2]
    right = crop[:, int(width * 0.55):]
    red = right[:, :, 0].astype(np.int16)
    green = right[:, :, 1].astype(np.int16)
    blue = right[:, :, 2].astype(np.int16)
    mask = (green >= 200) & (red >= 120) & (red <= 200) & (blue <= 160)
    return int(mask.sum()) > 200
