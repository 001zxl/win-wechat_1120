from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from .base_types import Badge
from .layout import Regions


def _red_mask(array: np.ndarray) -> np.ndarray:
    """微信未读红点：高饱和红（兼顾 3.x / 4.x 配色）。"""
    if array.ndim != 3:
        return np.zeros(array.shape[:2], dtype=bool)
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    return (red >= 170) & (green <= 110) & (blue <= 110) & ((red - green) >= 80)


def _components(mask: np.ndarray, min_pixels: int = 8) -> List[Tuple[int, int, int, int]]:
    """行游程连通域，避免引入 scipy。"""
    height, width = mask.shape
    parent: dict = {}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    runs: List[Tuple[int, int, int, int]] = []  # row, x1, x2, run_id
    for row in range(height):
        col = 0
        while col < width:
            if not mask[row, col]:
                col += 1
                continue
            start = col
            while col < width and mask[row, col]:
                col += 1
            run_id = len(runs)
            parent[run_id] = run_id
            runs.append((row, start, col - 1, run_id))
            # 与上一行的重叠 run 合并
            for prev in runs:
                if prev[0] != row - 1:
                    continue
                if prev[1] <= col - 1 and start <= prev[2]:
                    union(run_id, prev[3])

    groups: dict = {}
    for row, x1, x2, run_id in runs:
        root = find(run_id)
        entry = groups.setdefault(root, {"x1": x1, "y1": row, "x2": x2, "y2": row, "pixels": 0})
        entry["x1"] = min(entry["x1"], x1)
        entry["x2"] = max(entry["x2"], x2)
        entry["y2"] = max(entry["y2"], row)
        entry["pixels"] += x2 - x1 + 1

    boxes = []
    for entry in groups.values():
        if entry["pixels"] < min_pixels:
            continue
        w = entry["x2"] - entry["x1"] + 1
        h = entry["y2"] - entry["y1"] + 1
        if w > 60 or h > 40:  # 排除大片红色（图片/主题）
            continue
        if w < 4 or h < 4:
            continue
        boxes.append((entry["x1"], entry["y1"], entry["x2"], entry["y2"]))
    return boxes


def find_red_badges(image: Image.Image, regions: Regions, min_pixels: int = 8) -> List[Badge]:
    left, top, right, bottom = regions.session_list
    crop = np.array(image.crop((left, top, right, bottom)).convert("RGB"))
    mask = _red_mask(crop)
    badges = []
    for x1, y1, x2, y2 in _components(mask, min_pixels):
        badges.append(Badge(left + x1, top + y1, left + x2, top + y2))
    badges.sort(key=lambda item: item.y1)
    return badges


def click_point(badge: Badge, regions: Regions) -> Tuple[float, float]:
    """点在会话行文字区域，而不是红点本身。"""
    left, _top, right, _bottom = regions.session_list
    width = right - left
    x = left + width * 0.55
    _cx, cy = badge.center
    return x, cy
