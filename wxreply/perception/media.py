from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
from PIL import Image

from ..platform.base import MessageItem
from .layout import Regions
from .ocr import OcrLine


def _bands(height: int, band: int = 90) -> List[Tuple[int, int]]:
    return [(top, min(height, top + band)) for top in range(0, height, band)]


def detect_chat_images(image: Image.Image, regions: Regions, save_dir: str,
                       min_area: int = 12000) -> List[MessageItem]:
    """检测聊天区里的大块图片（如聊天截图），返回带 bbox 的图片消息。"""
    os.makedirs(save_dir, exist_ok=True)
    crop = np.array(image.crop(regions.messages).convert("RGB"), dtype=np.int16)
    height, width = crop.shape[:2]
    if height < 50 or width < 50:
        return []
    background = np.median(crop.reshape(-1, 3), axis=0)
    diff = np.abs(crop - background).sum(axis=2)
    mask = diff > 55
    row_counts = mask.sum(axis=1)
    rows = row_counts > max(12, width * 0.06)
    items: List[MessageItem] = []
    index = 0
    start = None
    for y in range(height + 1):
        active = y < height and bool(rows[y])
        if active and start is None:
            start = y
        elif not active and start is not None:
            top, bottom = start, y
            start = None
            if bottom - top < 80:
                continue
            sub = mask[top:bottom]
            cols = np.where(sub.any(axis=0))[0]
            if cols.size == 0:
                continue
            x1, x2 = int(cols.min()), int(cols.max())
            if x2 - x1 < 120:
                continue
            area = (bottom - top) * (x2 - x1)
            if area < min_area:
                continue
            piece = image.crop((regions.messages[0] + x1, regions.messages[1] + top,
                                regions.messages[0] + x2, regions.messages[1] + bottom))
            path = os.path.join(save_dir, f"chat-img-{int(time.time() * 1000)}-{index}.png")
            piece.save(path)
            items.append(MessageItem(
                kind="image", media_path=path, text="",
                bbox=(regions.messages[0] + x1, regions.messages[1] + top,
                      regions.messages[0] + x2, regions.messages[1] + bottom),
            ))
            index += 1
    return items


def extract_items(image: Image.Image, regions: Regions, ocr_lines: List[OcrLine],
                  save_dir: str, min_confidence: float = 0.5,
                  detect_media: bool = True) -> List[MessageItem]:
    """Windows 视觉兜底：文字气泡 + 疑似图片块。

    图片块用启发式：整条横带里有大量非背景像素，且没有任何 OCR 文本落在其中。
    真实微信上建议配合标定/训练的小模型；这里保证没有图片理解时至少不瞎猜。
    """
    chat_left, _top, chat_right, _bottom = regions.chat
    width = max(1, chat_right - chat_left)

    text_lines = [line for line in ocr_lines if line.score >= min_confidence and line.text.strip()]
    items: List[MessageItem] = []
    for line in text_lines:
        x_center, y_center = line.center
        direction = "out" if (x_center - chat_left) / width >= 0.55 else "in"
        items.append(MessageItem(kind="text", text=line.text.strip()))

    # 图片/媒体块检测（未启用视觉模型时关闭，避免误判）
    if not detect_media:
        return items
    messages = image.crop(regions.messages).convert("RGB")
    array = np.array(messages, dtype=np.int16)
    height = array.shape[0]
    background = np.median(array.reshape(-1, 3), axis=0)
    diff = np.abs(array - background).sum(axis=2)
    active = diff > 60
    line_centers = [line.center[1] for line in text_lines]
    os.makedirs(save_dir, exist_ok=True)
    for index, (top, bottom) in enumerate(_bands(height)):
        band = active[top:bottom]
        if band.size == 0:
            continue
        ratio = float(band.mean())
        if ratio < 0.06 or ratio > 0.85:
            continue
        if any(top <= center - regions.messages[1] <= bottom for center in line_centers):
            continue
        crop = messages.crop((0, top, messages.width, bottom))
        path = os.path.join(save_dir, f"media-{int(time.time() * 1000)}-{index}.png")
        crop.save(path)
        items.append(MessageItem(
            kind="image", media_path=path, text="",
            bbox=(regions.messages[0], regions.messages[1] + top,
                  regions.messages[2], regions.messages[1] + bottom),
        ))
    return items
